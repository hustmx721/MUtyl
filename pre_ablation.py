import copy
import gc
import os
import sys
import time
import warnings
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from evaluate import evaluate_acc_f1
from train import train_one_epoch
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, init_args, set_args, load_all
from utils.Logging import Logger
from utils.MULoader import Load_MU_Dataloader
from main_Dice import DiceLossConfig,_unpack_features_logits, compute_feature_mean, compute_unit_direction, \
    train_teacher, freeze_classifier_head

warnings.filterwarnings("ignore")

@dataclass
class AblationSpec:
    key: str
    name: str
    use_counterfactual: bool
    use_margin: bool
    use_subspace: bool
    use_retain: bool
    apply_projection: bool
    train_unlearn: bool

class ProjectedModel(nn.Module):
    def __init__(self, base_model: nn.Module, u_f: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("u_f", u_f)

    def forward(self, x):
        features, _ = _unpack_features_logits(self.base_model(x))
        coeff = torch.matmul(features, self.u_f)
        projected = features - coeff.unsqueeze(1) * self.u_f
        logits = self.base_model.clf(projected)
        return projected, logits

def select_counterfactual_logits(
    logits_r: torch.Tensor,
    y_r: torch.Tensor,
    y_f: torch.Tensor,
) -> torch.Tensor:
    device = logits_r.device
    batch_size = y_f.shape[0]
    indices = torch.empty(batch_size, dtype=torch.long, device=device)
    for idx, label in enumerate(y_f):
        candidates = torch.nonzero(y_r != label, as_tuple=False).flatten()
        if candidates.numel() == 0:
            indices[idx] = torch.randint(0, logits_r.shape[0], (1,), device=device)
        else:
            rand_idx = torch.randint(0, candidates.numel(), (1,), device=device)
            indices[idx] = candidates[rand_idx]
    return logits_r[indices]


def margin_suppression_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels.view(-1, 1), False)
    max_other = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
    return F.relu(true_logits - max_other + margin).mean()

def dice_unlearn_ablation(
    student: nn.Module,
    teacher: nn.Module,
    loaders: dict,
    u_f: torch.Tensor,
    config: DiceLossConfig,
    spec: AblationSpec,
    args,
    device: torch.device,
) -> nn.Module:
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    student.train()

    if args.dice_freeze_head:
        freeze_classifier_head(student)

    optimizer = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.dice_lr,
    )
    forget_loader = loaders["forget_train_loader"]
    retain_loader = loaders["remain_train_loader"]
    retain_cycle = cycle(retain_loader)

    forget_iters = args.dice_forget_iters or len(forget_loader)
    retain_iters = args.dice_retain_iters or len(retain_loader)

    kl_loss = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(args.dice_unlearn_epochs):
        student.train()
        last_loss_terms = None
        if spec.use_counterfactual or spec.use_margin or spec.use_subspace:
            for step, (x_f, y_f) in enumerate(forget_loader):
                if step >= forget_iters:
                    break
                x_f = x_f.to(device, non_blocking=True)
                y_f = y_f.to(device, non_blocking=True)

                loss_terms = []
                features_f, logits_f = _unpack_features_logits(student(x_f))

                if spec.use_counterfactual:
                    x_r, y_r = next(retain_cycle)
                    x_r = x_r.to(device, non_blocking=True)
                    y_r = y_r.to(device, non_blocking=True)
                    with torch.no_grad():
                        _, logits_r = _unpack_features_logits(teacher(x_r))
                        logits_r = select_counterfactual_logits(logits_r, y_r, y_f)
                        q = F.softmax(logits_r / config.temperature, dim=1)

                    log_p = F.log_softmax(logits_f / config.temperature, dim=1)
                    loss_terms.append(config.lambda_cf * kl_loss(log_p, q))

                if spec.use_margin:
                    loss_terms.append(
                        config.lambda_m * margin_suppression_loss(logits_f, y_f.long(), config.margin)
                    )

                if spec.use_subspace:
                    projection = torch.matmul(features_f, u_f)
                    loss_terms.append(config.lambda_sub * torch.mean(projection ** 2))

                if loss_terms:
                    loss_f = sum(loss_terms)
                    optimizer.zero_grad()
                    loss_f.backward()
                    optimizer.step()
                    last_loss_terms = loss_terms

        if spec.use_retain:
            student.train()
            for step, (x_r, y_r) in enumerate(retain_loader):
                if step >= retain_iters:
                    break
                x_r = x_r.to(device, non_blocking=True)
                y_r = y_r.to(device, non_blocking=True)

                _, logits_u = _unpack_features_logits(student(x_r))
                with torch.no_grad():
                    _, logits_t = _unpack_features_logits(teacher(x_r))
                    probs_t = F.softmax(logits_t / config.temperature, dim=1)

                log_probs_u = F.log_softmax(logits_u / config.temperature, dim=1)
                loss_kd = kl_loss(log_probs_u, probs_t)
                loss_r = config.beta_kd * loss_kd

                optimizer.zero_grad()
                loss_r.backward()
                optimizer.step()

        if (spec.use_counterfactual or spec.use_margin or spec.use_subspace) or spec.use_retain:
            print(
                f"Epoch {epoch + 1}/{args.dice_unlearn_epochs}: "
                f"Forget losses {[round(x.item(), 4) for x in last_loss_terms] if last_loss_terms else 'N/A'}"
            )

    return student


def compute_subspace_energy(model: nn.Module, loader, u_f: torch.Tensor, device: torch.device) -> float:
    model.eval()
    energies = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            features, _ = _unpack_features_logits(model(x))
            projection = torch.matmul(features, u_f)
            energies.append((projection ** 2).mean().item())
    if not energies:
        return float("nan")
    return float(np.mean(energies))


def compute_margin_stats(
    model: nn.Module,
    loader,
    margin: float,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    gaps = []
    hinge_hits = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            _, logits = _unpack_features_logits(model(x))
            true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, y.view(-1, 1), False)
            max_other = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
            gap = true_logits - max_other
            gaps.append(gap.mean().item())
            hinge_hits.append(((gap + margin) > 0).float().mean().item())
    if not gaps:
        return float("nan"), float("nan")
    return float(np.mean(hinge_hits)), float(np.mean(gaps))


def compute_entropy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    entropies = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            _, logits = _unpack_features_logits(model(x))
            probs = F.softmax(logits, dim=1)
            ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
            entropies.append(ent.mean().item())
    if not entropies:
        return float("nan")
    return float(np.mean(entropies))


def compute_kd_divergence(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    device: torch.device,
) -> float:
    student.eval()
    teacher.eval()
    kl_losses = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            _, logits_s = _unpack_features_logits(student(x))
            _, logits_t = _unpack_features_logits(teacher(x))
            log_probs_s = F.log_softmax(logits_s, dim=1)
            probs_t = F.softmax(logits_t, dim=1)
            kl = F.kl_div(log_probs_s, probs_t, reduction="batchmean")
            kl_losses.append(kl.item())
    if not kl_losses:
        return float("nan")
    return float(np.mean(kl_losses))


def build_ablation_specs():
    return {
        "G0": AblationSpec("G0", "Teacher", False, False, False, False, False, False),
        "G1": AblationSpec("G1", "R-only", False, False, False, True, False, True),
        "G2": AblationSpec("G2", "S-only", False, False, False, False, True, False),
        "G3": AblationSpec("G3", "M-only", False, True, False, False, False, True),
        "G4": AblationSpec("G4", "C-only", True, False, False, False, False, True),
        "G5": AblationSpec("G5", "C+M", True, True, False, False, False, True),
        "G6": AblationSpec("G6", "C+S", True, False, True, False, True, True),
        "G7": AblationSpec("G7", "M+S", False, True, True, False, True, True),
        "G8": AblationSpec("G8", "Full", True, True, True, True, True, True),
    }


def parse_groups(group_arg: str, specs: dict) -> list[AblationSpec]:
    if not group_arg:
        keys = list(specs.keys())
    else:
        keys = [k.strip() for k in group_arg.split(",") if k.strip()]
    selected = []
    for key in keys:
        if key not in specs:
            raise ValueError(f"Unknown ablation group: {key}")
        selected.append(specs[key])
    return selected


def main():
    args = init_args()
    args = set_args(args)
    apply_thread_limits(getattr(args, "torch_threads", 5))
    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")

    log_path = args.log_root / f"{args.dataset}_PreAblation_{args.model}.log"
    sys.stdout = Logger(log_path)

    seeds = list(range(args.seed, args.seed + args.repeats))
    if args.forget_subject is not None:
        forget_subjects = [args.forget_subject]
    elif args.dataset in ["MI", "SSVEP", "ERP"]:
        forget_subjects = list(range(0, 15))
    elif args.dataset in ["001", "004"]:
        forget_subjects = list(range(0, 9))
    else:
        forget_subjects = [None]

    specs = build_ablation_specs()
    group_arg = getattr(args, "ablation_groups", None)
    if not group_arg:
        group_arg = "G0,G1,G2,G3,G4,G5,G6,G7,G8"
    selected_specs = parse_groups(group_arg, specs)

    metric_cols = [
        "Retain_Acc",
        "Retain_F1",
        "Forget_Acc",
        "Forget_F1",
        "Energy_Forget",
        "Energy_Retain",
        "Margin_Act",
        "Margin_Gap",
        "Forget_Entropy",
        "KD_Retain",
    ]
    all_rows = []

    for forget_subject in forget_subjects:
        subject_rows = []
        print("=" * 30)
        print(f"Forget subject: {forget_subject}")
        for seed in seeds:
            args.seed = seed
            args.forget_subject = forget_subject
            args = set_args(args)
            start_time = time.time()
            print("=" * 30)
            print(f"dataset: {args.dataset}")
            print(f"model  : {args.model}")
            print(f"seed   : {args.seed}")
            print(f"gpu    : {args.gpuid}")
            print(f"is_task: {args.is_task}")

            set_seed(args.seed)
            loaders = Load_MU_Dataloader(
                seed=args.seed,
                dataset=args.dataset,
                batchsize=args.bs if args.dataset != "ERP" else 256,
                is_task=args.is_task,
                forget_subject=args.forget_subject,
            )
            print("=====================data are prepared===============")
            print(f"累计用时{time.time() - start_time:.4f}s!")
            print(f"Forget subject: {loaders['forget_subject']}")

            teacher = train_teacher(args, loaders["train_loader"], device)
            mu_f = compute_feature_mean(
                teacher,
                loaders["forget_train_loader"],
                device,
            )
            mu_r = compute_feature_mean(
                teacher,
                loaders["remain_train_loader"],
                device,
            )
            u_f = compute_unit_direction(mu_f, mu_r).to(device)

            for spec in selected_specs:
                print("-" * 20)
                print(f"Running {spec.key}: {spec.name}")
                if spec.key == "G0":
                    student = teacher
                elif spec.train_unlearn:
                    student = copy.deepcopy(teacher)
                    loss_config = DiceLossConfig(
                        temperature=args.dice_temperature,
                        margin=args.dice_margin,
                        lambda_cf=args.dice_lambda_cf if spec.use_counterfactual else 0.0,
                        lambda_m=args.dice_lambda_m if spec.use_margin else 0.0,
                        lambda_sub=args.dice_lambda_sub if spec.use_subspace else 0.0,
                        beta_kd=args.dice_beta_kd,
                    )
                    student = dice_unlearn_ablation(
                        student,
                        teacher,
                        loaders,
                        u_f,
                        loss_config,
                        spec,
                        args,
                        device,
                    )
                else:
                    student = teacher

                eval_model = ProjectedModel(student, u_f) if spec.apply_projection else student

                retain_acc, retain_f1 = evaluate_acc_f1(
                    eval_model, loaders["test_loader_remain"], args, device
                )
                forget_acc, forget_f1 = evaluate_acc_f1(
                    eval_model, loaders["test_loader_forget"], args, device
                )

                energy_f = compute_subspace_energy(student, loaders["test_loader_forget"], u_f, device)
                energy_r = compute_subspace_energy(student, loaders["test_loader_remain"], u_f, device)
                margin_act, margin_gap = compute_margin_stats(
                    student, loaders["test_loader_forget"], args.dice_margin, device
                )
                forget_entropy = compute_entropy(eval_model, loaders["test_loader_forget"], device)
                kd_retain = compute_kd_divergence(student, teacher, loaders["test_loader_remain"], device)

                subject_rows.append(
                    {
                        "Forget_Subject": loaders["forget_subject"],
                        "Seed": seed,
                        "Group": spec.key,
                        "Group_Name": spec.name,
                        "Retain_Acc": retain_acc,
                        "Retain_F1": retain_f1,
                        "Forget_Acc": forget_acc,
                        "Forget_F1": forget_f1,
                        "Energy_Forget": energy_f,
                        "Energy_Retain": energy_r,
                        "Margin_Act": margin_act,
                        "Margin_Gap": margin_gap,
                        "Forget_Entropy": forget_entropy,
                        "KD_Retain": kd_retain,
                    }
                )

                print(
                    "Retain Acc:{:.2f}% F1:{:.2f}% | Forget Acc:{:.2f}% F1:{:.2f}%".format(
                        retain_acc * 100,
                        retain_f1 * 100,
                        forget_acc * 100,
                        forget_f1 * 100,
                    )
                )
                print(
                    f"Energy_f:{energy_f:.4f} Energy_r:{energy_r:.4f} "
                    f"Margin_act:{margin_act:.4f} Margin_gap:{margin_gap:.4f} "
                    f"Entropy:{forget_entropy:.4f} KD:{kd_retain:.4f}"
                )

                model_path = args.model_root / f"{args.dataset}"
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                torch.save(
                    student.state_dict(),
                    model_path / f"PreAblation_{spec.key}_{args.model}_{args.seed}_forget{loaders['forget_subject']}.pth",
                )

                gc.collect()
                torch.cuda.empty_cache()

        subject_df = pd.DataFrame(subject_rows)
        subject_label = (
            subject_df["Forget_Subject"].iloc[0] if not subject_df.empty else forget_subject
        )
        avg_row = subject_df[metric_cols].mean(numeric_only=True)
        std_row = subject_df[metric_cols].std(numeric_only=True)
        subject_rows.append(
            {
                "Forget_Subject": subject_label,
                "Seed": "AVG",
                "Group": "ALL",
                "Group_Name": "Average",
                **avg_row.to_dict(),
            }
        )
        subject_rows.append(
            {
                "Forget_Subject": subject_label,
                "Seed": "STD",
                "Group": "ALL",
                "Group_Name": "Std",
                **std_row.to_dict(),
            }
        )
        all_rows.extend(subject_rows)

    df = pd.DataFrame(all_rows)
    df = df.round(4)
    csv_path = args.csv_root / f"{args.dataset}"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv(csv_path / f"PreAblation_{args.model}.csv", index=False)


if __name__ == "__main__":
    main()
