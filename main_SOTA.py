import argparse
import copy
import gc
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluate import evaluate_acc_f1
from train import train_one_epoch
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, load_all, set_args
from utils.Logging import Logger
from utils.MULoader import Load_MU_Dataloader

from SOTA.DELETE import DELETE
from SOTA.ESC import ESC
from SOTA.GA import GA, GAConfig
from SOTA.LAF import LAF, LAFConfig
from SOTA.SCRUB import SCRUB, SCRUBConfig
from SOTA.SISA import SISA, SISAConfig

warnings.filterwarnings("ignore")


SUPPORTED_METHODS = ["ESC", "DELETE", "SCRUB", "GA", "LAF", "SISA"]


def parse_args():
    project_root = Path(__file__).resolve().parent
    default_log_root = project_root / "logs"
    default_model_root = project_root / "ModelSave"
    default_csv_root = project_root / "csv"

    parser = argparse.ArgumentParser(description="Batch runner for SOTA unlearning baselines")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--gpuid", type=int, default=9)
    parser.add_argument("--nclass", type=int, default=9)
    parser.add_argument("--channel", type=int, default=22)
    parser.add_argument("--timepoint", type=int, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--initlr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--earlystop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--model", type=str, default="EEGNet")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--is_task", type=bool, default=True)
    parser.add_argument("--torch_threads", type=int, default=5)
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs to evaluate (default: all available subjects)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(SUPPORTED_METHODS),
        help=f"Comma-separated list of methods to run (default: {','.join(SUPPORTED_METHODS)})",
    )
    parser.add_argument("--log_root", type=Path, default=default_log_root)
    parser.add_argument("--model_root", type=Path, default=default_model_root)
    parser.add_argument("--csv_root", type=Path, default=default_csv_root)
    return parser.parse_args()


def _parse_subjects(subjects_arg, available_subjects):
    if subjects_arg:
        subject_list = [s for s in subjects_arg.split(",") if s.strip()]
        return [int(s) for s in subject_list]
    if available_subjects is None or len(available_subjects) == 0:
        return []
    return [int(s) for s in np.unique(available_subjects)]


def _unpack_features_logits(output):
    if isinstance(output, (tuple, list)):
        if len(output) != 2:
            raise ValueError("Expected model to return (features, logits).")
        return output
    raise ValueError("Expected model to return (features, logits).")


def _train_teacher(args, train_loader, device: torch.device) -> nn.Module:
    model, optimizer, _ = load_all(args)
    model.to(device)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)

    clf_loss_func = nn.CrossEntropyLoss().to(device)
    for _ in range(args.epoch):
        train_one_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            clf_loss_func=clf_loss_func,
        )
    return model


@dataclass
class LinearVAEConfig:
    input_dim: int
    latent_dim: int = 64
    hidden_dim: int = 128


class LinearVAE(nn.Module):
    def __init__(self, config: LinearVAEConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        z = mu + sigma * torch.randn_like(sigma)
        return self.decoder(z), z, mu, sigma


def _infer_feature_dim(model: nn.Module, loader, device: torch.device) -> int:
    model.eval()
    with torch.no_grad():
        sample_x, _ = next(iter(loader))
        sample_x = sample_x.to(device, non_blocking=True)
        features, _ = _unpack_features_logits(model(sample_x))
        return features.shape[-1]


class SISAEnsemble(nn.Module):
    def __init__(self, shard_models):
        super().__init__()
        self.shard_models = nn.ModuleList(shard_models)

    def forward(self, x):
        logits_list = []
        for model in self.shard_models:
            output = model(x)
            if isinstance(output, (tuple, list)):
                logits = output[1]
            else:
                logits = output
            logits_list.append(logits)
        logits_stack = torch.stack(logits_list, dim=0)
        logits_mean = logits_stack.mean(dim=0)
        return logits_mean, logits_mean


def _save_model(model: nn.Module, args, method: str, seed: int, forget_subject: int):
    model_path = args.model_root / f"{args.dataset}"
    os.makedirs(model_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        model_path / f"{method}_{args.model}_seed{seed}_forget{forget_subject}.pth",
    )


def _save_sisa_models(models, args, seed: int, forget_subject: int):
    model_path = args.model_root / f"{args.dataset}"
    os.makedirs(model_path, exist_ok=True)
    for idx, model in enumerate(models):
        torch.save(
            model.state_dict(),
            model_path / f"SISA_{args.model}_seed{seed}_forget{forget_subject}_shard{idx}.pth",
        )


def _run_method(method: str, args, loaders, device: torch.device):
    teacher = _train_teacher(args, loaders["train_loader"], device)
    if method == "ESC":
        model = copy.deepcopy(teacher)
        esc = ESC(device=device)
        model = esc.unlearn(model, loaders["forget_train_loader"])
        return model
    if method == "DELETE":
        model = copy.deepcopy(teacher)
        delete = DELETE(device=device)
        model = delete.unlearn(model, loaders["forget_train_loader"])
        return model
    if method == "SCRUB":
        model = copy.deepcopy(teacher)
        scrub = SCRUB(SCRUBConfig(device=device))
        model = scrub.unlearn(
            model,
            {
                "forget": loaders["forget_train_loader"],
                "retain": loaders["remain_train_loader"],
            },
        )
        return model
    if method == "GA":
        model = copy.deepcopy(teacher)
        ga = GA(GAConfig(device=device))
        model = ga.unlearn(
            model,
            {
                "forget": loaders["forget_train_loader"],
                "retain": loaders["remain_train_loader"],
            },
        )
        return model
    if method == "LAF":
        model = copy.deepcopy(teacher)
        laf = LAF(LAFConfig(device=device))
        feature_dim = _infer_feature_dim(model, loaders["remain_train_loader"], device)
        vae_config = LinearVAEConfig(input_dim=feature_dim)
        s_vae = LinearVAE(vae_config)
        u_vae = LinearVAE(vae_config)
        model, _ = laf.unlearn(
            model,
            {
                "unlearn": loaders["forget_train_loader"],
                "remain": loaders["remain_train_loader"],
            },
            s_vae,
            u_vae,
            train_vae=True,
        )
        return model
    if method == "SISA":
        sisa = SISA(SISAConfig(device=device))

        def _model_factory():
            model, _, _ = load_all(args)
            return model

        shard_loaders = [
            loaders["remain_train_loader"],
            loaders["forget_train_loader"],
        ]
        shard_models = sisa.train_shards(_model_factory, shard_loaders)
        retrain_shards = [
            loaders["remain_train_loader"],
            loaders["remain_train_loader"],
        ]
        shard_models = sisa.unlearn(
            shard_models,
            _model_factory,
            retrain_shards,
            forget_shards=[1],
        )
        return SISAEnsemble(shard_models)
    raise ValueError(f"Unknown method: {method}")


def main():
    args = parse_args()
    apply_thread_limits(getattr(args, "torch_threads", 4))
    args.is_task = True
    args = set_args(args)

    log_path = args.log_root / f"SOTA_{args.dataset}_{args.model}.log"
    sys.stdout = Logger(log_path)

    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")
    seeds = list(range(args.seed, args.seed + args.repeats))

    method_list = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in method_list:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method '{method}'. Supported: {SUPPORTED_METHODS}")

    initial_loaders = Load_MU_Dataloader(
        args.seed,
        args.dataset,
        batchsize=args.bs,
        is_task=args.is_task,
    )
    available_subjects = initial_loaders.get("available_subjects")
    subjects = _parse_subjects(args.subjects, available_subjects)
    if not subjects:
        raise ValueError("No subjects available for SOTA evaluation.")

    metric_cols = [
        "Retain_Acc",
        "Retain_F1",
        "Forget_Acc",
        "Forget_F1",
        "Runtime_Sec",
    ]

    for method in method_list:
        all_rows = []
        for subject_id in subjects:
            subject_rows = []
            print("=" * 30)
            print(f"Method: {method} | Forget subject: {subject_id}")

            for seed in seeds:
                args.seed = seed
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
                    forget_subject=subject_id,
                )
                print("=====================data are prepared===============")
                print(f"Forget subject: {loaders['forget_subject']}")

                model = _run_method(method, args, loaders, device)

                if method == "SISA":
                    _save_sisa_models(model.shard_models, args, seed, loaders["forget_subject"])
                else:
                    _save_model(model, args, method, seed, loaders["forget_subject"])

                retain_acc, retain_f1 = evaluate_acc_f1(
                    model, loaders["test_loader_remain"], args, device
                )
                forget_acc, forget_f1 = evaluate_acc_f1(
                    model, loaders["test_loader_forget"], args, device
                )

                runtime = time.time() - start_time
                subject_rows.append(
                    {
                        "Method": method,
                        "Forget_Subject": loaders["forget_subject"],
                        "Seed": seed,
                        "Retain_Acc": retain_acc,
                        "Retain_F1": retain_f1,
                        "Forget_Acc": forget_acc,
                        "Forget_F1": forget_f1,
                        "Runtime_Sec": runtime,
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
                print(f"累计用时{runtime:.4f}s!")
                gc.collect()
                torch.cuda.empty_cache()

            subject_df = pd.DataFrame(subject_rows)
            subject_label = (
                subject_df["Forget_Subject"].iloc[0]
                if not subject_df.empty
                else subject_id
            )
            avg_row = subject_df[metric_cols].mean()
            std_row = subject_df[metric_cols].std()
            subject_rows.append(
                {
                    "Method": method,
                    "Forget_Subject": subject_label,
                    "Seed": "AVG",
                    **avg_row.to_dict(),
                }
            )
            subject_rows.append(
                {
                    "Method": method,
                    "Forget_Subject": subject_label,
                    "Seed": "STD",
                    **std_row.to_dict(),
                }
            )
            all_rows.extend(subject_rows)

        df = pd.DataFrame(all_rows, columns=["Method", "Forget_Subject", "Seed"] + metric_cols)
        df = df.round(4)
        csv_path = args.csv_root / f"{args.dataset}"
        os.makedirs(csv_path, exist_ok=True)
        df.to_csv(csv_path / f"SOTA_{method}_{args.model}.csv", index=False)


if __name__ == "__main__":
    main()
