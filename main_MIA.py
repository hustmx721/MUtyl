import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from utils.MIA import run_mia_from_mu_loaders
from utils.MULoader import Load_MU_Dataloader
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, set_args
from utils.models import LoadModel


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser("Evaluate DiCE checkpoints with black-box MIA")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="EEGNet")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--is_task", type=str2bool, default=True)
    parser.add_argument(
        "--forget_subject",
        type=int,
        default=None,
        help="指定单个被遗忘受试者；不指定时会遍历该数据集所有受试者",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="可选：手动指定多个受试者ID，例如 '0,3,5'（优先级高于 --forget_subject）",
    )
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="可选：直接指定单个checkpoint路径")
    parser.add_argument("--model_root", type=Path, default=project_root / "ModelSave")
    parser.add_argument("--csv_root", type=Path, default=project_root / "csv")
    parser.add_argument("--torch_threads", type=int, default=5)
    parser.add_argument(
        "--mia_methods",
        type=str,
        default="correctness,confidence,entropy,modified_entropy",
        help="逗号分隔的MIA方法列表",
    )
    return parser.parse_args()


def resolve_forget_subjects(dataset: str, user_forget_subject: int | None, subjects_arg: str | None):
    if subjects_arg:
        return [int(x.strip()) for x in subjects_arg.split(",") if x.strip()]
    if user_forget_subject is not None:
        return [user_forget_subject]
    if dataset in ["MI", "SSVEP", "ERP"]:
        return list(range(0, 15))
    if dataset in ["001", "004"]:
        return list(range(0, 9))
    return [None]


def load_checkpoint(model, ckpt_path: Path, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


def main():
    args = parse_args()
    args = set_args(args)
    apply_thread_limits(args.torch_threads)

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
    methods = [m.strip() for m in args.mia_methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("--mia_methods 不能为空")

    seeds = list(range(args.seed, args.seed + args.repeats))
    forget_subjects = resolve_forget_subjects(args.dataset, args.forget_subject, args.subjects)

    rows = []
    for forget_subject in forget_subjects:
        for seed in seeds:
            set_seed(seed)
            loaders = Load_MU_Dataloader(
                seed=seed,
                dataset=args.dataset,
                batchsize=args.bs if args.dataset != "ERP" else 256,
                is_task=args.is_task,
                forget_subject=forget_subject,
            )
            resolved_forget_subject = loaders["forget_subject"]

            model = LoadModel(
                model_name=args.model,
                Chans=args.channel,
                Samples=int(args.fs * args.timepoint),
                n_classes=args.nclass,
            ).to(device)
            model.eval()

            if args.checkpoint is not None:
                ckpt_path = args.checkpoint
            else:
                ckpt_path = (
                    args.model_root
                    / f"{args.dataset}"
                    / f"DiCE_{args.model}_{seed}_forget{resolved_forget_subject}.pth"
                )
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            load_checkpoint(model, ckpt_path, device)
            mia_result = run_mia_from_mu_loaders(loaders, model, device=device, methods=methods)

            row = {
                "dataset": args.dataset,
                "model": args.model,
                "seed": seed,
                "forget_subject": resolved_forget_subject,
                "checkpoint": str(ckpt_path),
            }
            for method, summary in mia_result.items():
                row[f"{method}_attack_acc"] = summary.attack_acc
                row[f"{method}_member_acc"] = summary.train_acc
                row[f"{method}_non_member_acc"] = summary.test_acc
            rows.append(row)

            print("=" * 80)
            print(json.dumps(row, indent=2, ensure_ascii=False))

    result_df = pd.DataFrame(rows)
    out_dir = args.csv_root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"DiCE_MIA_{args.model}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"MIA结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
