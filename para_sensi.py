import copy
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

from evaluate import evaluate_acc_f1
from main_Dice import DiceLossConfig, compute_feature_mean, compute_unit_direction, dice_unlearn, train_teacher
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, build_arg_parser, set_args
from utils.Logging import Logger
from utils.MULoader import Load_MU_Dataloader

warnings.filterwarnings("ignore")


def _parse_values(raw: str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
            continue
        except ValueError:
            pass
        try:
            values.append(float(item))
            continue
        except ValueError:
            pass
        values.append(item)
    if not values:
        raise ValueError("No valid sensitivity values provided.")
    return values


def _resolve_forget_subjects(args):
    if args.forget_subject is not None:
        return [args.forget_subject]
    if args.dataset in ["MI", "SSVEP", "ERP"]:
        return list(range(0, 15))
    if args.dataset in ["001", "004"]:
        return list(range(0, 9))
    return [None]


def run_sensitivity(
    base_args,
    device: torch.device,
    param_name: str,
    values: list,
):
    seeds = list(range(base_args.seed, base_args.seed + base_args.repeats))
    forget_subjects = _resolve_forget_subjects(base_args)

    rows = []
    for value in values:
        args = copy.deepcopy(base_args)
        setattr(args, param_name, value)
        print("=" * 30)
        print(f"Sensitivity param: {param_name} -> {value}")
        for forget_subject in forget_subjects:
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
                print(f"{param_name}: {getattr(args, param_name)}")

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

                retain_acc, retain_f1 = evaluate_acc_f1(
                    student, loaders["test_loader_remain"], args, device
                )
                forget_acc, forget_f1 = evaluate_acc_f1(
                    student, loaders["test_loader_forget"], args, device
                )

                rows.append(
                    {
                        "Param": param_name,
                        "Value": value,
                        "Forget_Subject": loaders["forget_subject"],
                        "Seed": seed,
                        "Retain_Acc": retain_acc,
                        "Retain_F1": retain_f1,
                        "Forget_Acc": forget_acc,
                        "Forget_F1": forget_f1,
                    }
                )
    return rows


def main():
    parser = build_arg_parser()
    parser.set_defaults(
        dice_lambda_cf=3.0,
        dice_lambda_m=0.15,
        dice_lambda_sub=1.0,
        dice_temperature=2.0,
    )
    parser.add_argument(
        "--sensi_values",
        type=str,
        default="1.0,2.0,3.0",
        help="Comma-separated list of values to sweep.",
    )
    parser.add_argument(
        "--sensi_params",
        type=str,
        default="dice_lambda_cf,dice_lambda_m,dice_lambda_sub,dice_temperature",
        help="Comma-separated list of parameters to sweep.",
    )
    parser.add_argument(
        "--temperature_values",
        type=str,
        default="1, 2, 3, 4, 5",
        help="Values for dice_temperature.",
    )
    parser.add_argument(
        "--lambda_cf_values",
        type=str,
        default="1, 2, 3, 4, 5",
        help="Values for dice_lambda_cf.",
    )
    parser.add_argument(
        "--lambda_m_values",
        type=str,
        default="0.10, 0.15, 0.20, 0.25, 0.30",
        help="Values for dice_lambda_m.",
    )
    parser.add_argument(
        "--lambda_sub_values",
        type=str,
        default="1, 2, 3, 4, 5",
        help="Values for dice_lambda_sub.",
    )
    args = parser.parse_args()
    args = set_args(args)
    apply_thread_limits(getattr(args, "torch_threads", 5))
    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")
    base_args = copy.deepcopy(args)

    log_path = args.log_root / f"{args.dataset}_sensi_{args.model}.log"
    sys.stdout = Logger(log_path)

    param_list = [name.strip() for name in args.sensi_params.split(",") if name.strip()]
    values_map = {
        "dice_temperature": _parse_values(args.temperature_values),
        "dice_lambda_cf": _parse_values(args.lambda_cf_values),
        "dice_lambda_m": _parse_values(args.lambda_m_values),
        "dice_lambda_sub": _parse_values(args.lambda_sub_values),
    }

    csv_path = args.csv_root / f"{args.dataset}"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for param_name in param_list:
        sensi_values = values_map.get(param_name)
        if not sensi_values:
            raise ValueError(f"No sensitivity values configured for {param_name}.")
        rows = run_sensitivity(base_args, device, param_name, sensi_values)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        summary = (
            df.groupby(["Param", "Value", "Forget_Subject"], as_index=False)
            .agg(
                Retain_Acc=("Retain_Acc", "mean"),
                Retain_F1=("Retain_F1", "mean"),
                Forget_Acc=("Forget_Acc", "mean"),
                Forget_F1=("Forget_F1", "mean"),
                Seed_Count=("Seed", "count"),
            )
            .round(4)
        )
        summary.to_csv(
            csv_path / f"DiCE_sensi_{args.model}_{param_name}.csv",
            index=False,
        )
        print(f"Saved sensitivity results to {csv_path}")


if __name__ == "__main__":
    main()
