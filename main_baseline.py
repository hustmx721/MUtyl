import argparse
import gc
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from evaluate import evaluate_acc_f1, evaluate
from train import train_one_epoch
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, load_all, set_args
from utils.Logging import Logger
from utils.MULoader import Load_MU_Dataloader

warnings.filterwarnings("ignore")


def parse_args():
    project_root = Path(__file__).resolve().parent
    default_log_root = project_root / "logs"
    default_model_root = project_root / "ModelSave"
    default_csv_root = project_root / "csv"

    parser = argparse.ArgumentParser(description="Clean-data baseline for task classification")
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


def train_baseline(train_loader, val_loader, args, save_prefix):
    print("-" * 20 + "开始训练!" + "-" * 20)

    model, optimizer, device = load_all(args)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)

    clf_loss_func = nn.CrossEntropyLoss().to(device)

    best_epoch = 0
    best_acc = 0

    for epoch in tqdm(range(args.epoch), desc="Training:"):
        train_loss, train_acc, train_f1, train_bca, train_eer = train_one_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            clf_loss_func=clf_loss_func,
        )

        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate(
            model=model,
            dataloader=val_loader,
            args=args,
            device=device,
        )

        if (epoch - best_epoch) > args.earlystop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model_path = args.model_root / f"{args.dataset}"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), model_path / f"{save_prefix}.pth")

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch:{epoch+1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}\tTrain_loss:{train_loss:.6f}\tVal_loss:{val_loss:.6f}"
            )
            print(
                f"  Train_F1:{train_f1:.4f}, BCA:{train_bca:.4f}, EER:{train_eer:.4f} | Val_F1:{val_f1:.4f}, BCA:{val_bca:.4f}, EER:{val_eer:.4f}"
            )

    print("-" * 20 + "训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    return model


def main():
    args = parse_args()
    apply_thread_limits(getattr(args, "torch_threads", 4))
    args.is_task = True
    args = set_args(args)

    log_path = args.log_root / f"{args.dataset}_baseline_{args.model}.log"
    sys.stdout = Logger(log_path)

    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")
    seeds = list(range(args.seed, args.seed + args.repeats))

    initial_loaders = Load_MU_Dataloader(
        args.seed,
        args.dataset,
        batchsize=args.bs,
        is_task=args.is_task,
    )
    available_subjects = initial_loaders.get("available_subjects")
    subjects = _parse_subjects(args.subjects, available_subjects)
    if not subjects:
        raise ValueError("No subjects available for baseline evaluation.")

    all_rows = []
    metric_cols = ["Retain_Acc", "Retain_F1", "Forget_Acc", "Forget_F1"]
    subject_rows_map = {subject_id: [] for subject_id in subjects}

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
        train_loaders = Load_MU_Dataloader(
            args.seed,
            args.dataset,
            batchsize=args.bs,
            is_task=args.is_task,
            forget_subject=subjects[0],
        )
        print("=====================data are prepared===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")

        save_prefix = f"Baseline_{args.model}_seed{args.seed}_allsubjects"
        model = train_baseline(
            train_loaders["train_loader"],
            train_loaders["test_loader_remain"],
            args,
            save_prefix,
        )

        for subject_id in subjects:
            eval_loaders = Load_MU_Dataloader(
                args.seed,
                args.dataset,
                batchsize=args.bs,
                is_task=args.is_task,
                forget_subject=subject_id,
            )
            retain_acc, retain_f1 = evaluate_acc_f1(
                model, eval_loaders["test_loader_remain"], args, device
            )
            forget_acc, forget_f1 = evaluate_acc_f1(
                model, eval_loaders["test_loader_forget"], args, device
            )

            subject_rows_map[subject_id].append(
                {
                    "Forget_Subject": eval_loaders["forget_subject"],
                    "Seed": seed,
                    "Retain_Acc": retain_acc,
                    "Retain_F1": retain_f1,
                    "Forget_Acc": forget_acc,
                    "Forget_F1": forget_f1,
                }
            )
            print(
                "Subject {} | Retain Acc:{:.2f}% F1:{:.2f}% | Forget Acc:{:.2f}% F1:{:.2f}%".format(
                    subject_id,
                    retain_acc * 100,
                    retain_f1 * 100,
                    forget_acc * 100,
                    forget_f1 * 100,
                )
            )

        print("=====================baseline done===================")
        print(f"累计用时{time.time() - start_time:.4f}s!")
        gc.collect()
        torch.cuda.empty_cache()

    for subject_id, subject_rows in subject_rows_map.items():
        subject_df = pd.DataFrame(subject_rows)
        subject_label = (
            subject_df["Forget_Subject"].iloc[0] if not subject_df.empty else subject_id
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

    result_df = pd.DataFrame(all_rows)
    csv_path = args.csv_root / f"{args.dataset}"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    result_df.to_csv(csv_path / f"Baseline_{args.model}.csv", index=False)


if __name__ == "__main__":
    main()
