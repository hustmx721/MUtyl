import copy
import os
import sys
import time
import gc
import warnings
from dataclasses import dataclass
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent)
from evaluate import evaluate_acc_f1, calculate_acc_f1
from train import train_one_epoch
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, init_args, set_args, load_all
from utils.Logging import Logger
from utils.MULoader import Load_MU_Dataloader

warnings.filterwarnings("ignore")


@dataclass
class DiceLossConfig:
    temperature: float
    margin: float
    lambda_cf: float
    lambda_m: float
    lambda_sub: float
    beta_kd: float


def _unpack_features_logits(output):
    if isinstance(output, (tuple, list)):
        if len(output) != 2:
            raise ValueError("Expected model to return (features, logits).")
        return output
    raise ValueError("Expected model to return (features, logits).")


def compute_feature_mean(
    model: nn.Module,
    loader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    feature_sum = None
    count = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            features, _ = _unpack_features_logits(model(x))
            batch_features = features.detach()
            if feature_sum is None:
                feature_sum = torch.zeros(batch_features.shape[1], device=device)
            feature_sum += batch_features.sum(dim=0)
            count += batch_features.shape[0]
    if count == 0:
        raise ValueError("No samples available for feature mean computation.")
    return feature_sum / count


def compute_unit_direction(mu_f: torch.Tensor, mu_r: torch.Tensor) -> torch.Tensor:
    diff = mu_f - mu_r
    norm = torch.norm(diff, p=2)
    if norm == 0:
        return torch.zeros_like(diff)
    return diff / norm


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


def train_teacher(
    args,
    train_loader,
    device: torch.device,
) -> nn.Module:
    model, optimizer, _ = load_all(args)
    model.to(device)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)

    clf_loss_func = nn.CrossEntropyLoss().to(device)
    for epoch in tqdm(range(args.dice_teacher_epochs), desc="Teacher Training:"):
        train_one_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            clf_loss_func=clf_loss_func,
        )
    return model


def freeze_classifier_head(model: nn.Module) -> None:
    clf = model.clf
    for param in clf.parameters():
        param.requires_grad = False


def dice_unlearn(
    student: nn.Module,
    teacher: nn.Module,
    loaders: dict,
    u_f: torch.Tensor,
    config: DiceLossConfig,
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
        for step, (x_f, y_f) in enumerate(forget_loader):
            if step >= forget_iters:
                break
            x_r, y_r = next(retain_cycle)
            x_f = x_f.to(device, non_blocking=True)
            y_f = y_f.to(device, non_blocking=True)
            x_r = x_r.to(device, non_blocking=True)
            y_r = y_r.to(device, non_blocking=True)

            with torch.no_grad():
                _, logits_r = _unpack_features_logits(teacher(x_r))
                logits_r = select_counterfactual_logits(logits_r, y_r, y_f)
                q = F.softmax(logits_r / config.temperature, dim=1)

            features_f, logits_f = _unpack_features_logits(student(x_f))
            log_p = F.log_softmax(logits_f / config.temperature, dim=1)

            loss_cf = kl_loss(log_p, q)
            loss_marg = margin_suppression_loss(logits_f, y_f.long(), config.margin)
            projection = torch.matmul(features_f, u_f)
            loss_sub = torch.mean(projection ** 2)
            loss_f = (
                config.lambda_cf * loss_cf
                + config.lambda_m * loss_marg
                + config.lambda_sub * loss_sub
            )

            optimizer.zero_grad()
            loss_f.backward()
            optimizer.step()

        student.train()
        for step, (x_r, y_r) in enumerate(retain_loader):
            if step >= retain_iters:
                break
            x_r = x_r.to(device, non_blocking=True)
            y_r = y_r.to(device, non_blocking=True)

            _, logits_u = _unpack_features_logits(student(x_r))
            # loss_ce = F.cross_entropy(logits_u, y_r)

            with torch.no_grad():
                _, logits_t = _unpack_features_logits(teacher(x_r))
                probs_t = F.softmax(logits_t / config.temperature, dim=1)

            log_probs_u = F.log_softmax(logits_u / config.temperature, dim=1)
            loss_kd = kl_loss(log_probs_u, probs_t)
            loss_r = config.beta_kd * loss_kd

            optimizer.zero_grad()
            loss_r.backward()
            optimizer.step()

        print(
            f"Epoch {epoch + 1}/{args.dice_unlearn_epochs}: "
            f"Forget loss {loss_f.item():.4f} | Retain loss {loss_r.item():.4f}"
        )

    return student


def main():
    args = init_args()
    args = set_args(args)
    apply_thread_limits(getattr(args, "torch_threads", 5))
    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")

    log_path = args.log_root / f"{args.dataset}_DiCE_{args.model}.log"
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

    metric_cols = [
        "Retain_Acc",
        "Retain_F1",
        "Forget_Acc",
        "Forget_F1",
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
                args.seed,
                args.dataset,
                batchsize=args.bs,
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

            student = copy.deepcopy(teacher)
            loss_config = DiceLossConfig(
                temperature=args.dice_temperature,
                margin=args.dice_margin,
                lambda_cf=args.dice_lambda_cf,
                lambda_m=args.dice_lambda_m,
                lambda_sub=args.dice_lambda_sub,
                beta_kd=args.dice_beta_kd,
            )
            student = dice_unlearn(student, teacher, loaders, u_f, loss_config, args, device)

            model_path = args.model_root / f"{args.dataset}"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(
                student.state_dict(),
                model_path / f"DiCE_{args.model}_{args.seed}_forget{loaders['forget_subject']}.pth",
            )

            retain_acc, retain_f1 = evaluate_acc_f1(
                student, loaders["test_loader_remain"], args, device
            )
            forget_acc, forget_f1 = evaluate_acc_f1(
                student, loaders["test_loader_forget"], args, device
            )

            subject_rows.append(
                {
                    "Forget_Subject": loaders["forget_subject"],
                    "Seed": seed,
                    "Retain_Acc": retain_acc,
                    "Retain_F1": retain_f1,
                    "Forget_Acc": forget_acc,
                    "Forget_F1": forget_f1,
                }
            )
            print(
                "Retain Test  Acc:{:.2f}% F1:{:.2f}%".format(
                    retain_acc * 100,
                    retain_f1 * 100,
                )
            )
            print(
                "Forget Test  Acc:{:.2f}% F1:{:.2f}%".format(
                    forget_acc * 100,
                    forget_f1 * 100,
                )
            )
            print("=====================unlearning done===================")
            print(f"累计用时{time.time() - start_time:.4f}s!")
            gc.collect()
            torch.cuda.empty_cache()

        subject_df = pd.DataFrame(subject_rows)
        subject_label = (
            subject_df["Forget_Subject"].iloc[0] if not subject_df.empty else forget_subject
        )
        avg_row = subject_df[metric_cols].mean()
        std_row = subject_df[metric_cols].std()
        subject_rows.append(
            {
                "Forget_Subject": subject_label,
                "Seed": "AVG",
                **avg_row.to_dict(),
            }
        )
        subject_rows.append(
            {
                "Forget_Subject": subject_label,
                "Seed": "STD",
                **std_row.to_dict(),
            }
        )
        all_rows.extend(subject_rows)

    df = pd.DataFrame(all_rows, columns=["Forget_Subject", "Seed"] + metric_cols)
    df = df.round(4)
    csv_path = args.csv_root / f"{args.dataset}"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv(csv_path / f"DiCE_{args.model}.csv")


if __name__ == "__main__":
    main()
