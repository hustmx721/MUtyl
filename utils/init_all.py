import sys
import os
from pathlib import Path

import torch
import argparse
from .data_loader import *
from .models import LoadModel


def init_args():
    project_root = Path(__file__).resolve().parent.parent
    default_log_root = project_root / "logs"
    default_model_root = project_root / "ModelSave"
    default_csv_root = project_root / "csv"
    default_sys_path = project_root

    parser = argparse.ArgumentParser(description="Model Train Hyperparameter")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--gpuid", type=int, default=9)
    parser.add_argument("--nclass", type=int, default=9)  # 用户数量
    parser.add_argument("--channel", type=int, default=22)
    parser.add_argument("--timepoint", type=int, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--initlr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--earlystop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--model", type=str, default="EEGNet")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--is_task", type=bool, default=True)
    parser.add_argument("--torch_threads", type=int, default=4,
                        help="Number of threads to use for torch operations")
    # logs path
    parser.add_argument("--log_root", type=Path, default=default_log_root,
                        help="Directory to store training logs")
    parser.add_argument("--model_root", type=Path, default=default_model_root,
                        help="Directory to store trained model checkpoints")
    parser.add_argument("--csv_root", type=Path, default=default_csv_root,
                        help="Directory to store exported CSV results")
    parser.add_argument("--extra_sys_path", type=Path, default=default_sys_path,
                        help="Additional path to append to sys.path for imports")
    args = parser.parse_args()

    # Append additional sys.path if provided
    if args.extra_sys_path:
        extra_path = args.extra_sys_path
        if not extra_path.is_absolute():
            extra_path = project_root / extra_path
        resolved_path = extra_path.resolve()
        if str(resolved_path) not in sys.path:
            sys.path.append(str(resolved_path))

    return args


def set_args(args: argparse.ArgumentParser):
    OpenBMI = ["MI", "SSVEP", "ERP"]
    M3CV = ["Rest", "Transient", "Steady", "P300", "Motor", "SSVEP_SA"]
    if args.dataset in OpenBMI:
        args.channel = 62
        args.fs = 250
        if args.dataset == "ERP":
            args.nclass = 2
            args.timepoint = 0.8
        elif args.dataset == "MI":
            args.nclass = 2
            args.timepoint = 4
        elif args.dataset == "SSVEP":
            args.nclass = 4
            args.timepoint = 4
        # UID分类
        if not args.is_task:
            args.nclass = 54
    elif args.dataset in M3CV:
        args.channel = 64
        args.fs = 250
        args.timepoint = 4
        match args.dataset:
            case "Rest":
                args.nclass = 2
            case "Transient":
                args.nclass = 3
            case "Steady":
                args.nclass = 3
            case "Motor":
                args.nclass = 3
        # UID分类
        if not args.is_task:
            args.nclass = 20
    elif args.dataset in ["001", "004"]:
        args.channel = 22 if args.dataset == "001" else 3
        args.nclass = 4 if args.dataset == "001" else 2
        args.fs = 250
        args.timepoint = 4
        # UID分类
        if not args.is_task:
            args.nclass = 9
    return args


def load_data(args: argparse.ArgumentParser):
    batch_size = getattr(args, "bs", 64)
    return Load_Dataloader(
        args.seed,
        args.dataset,
        batchsize=batch_size,
        is_task=args.is_task,
    )



def apply_thread_limits(thread_count: int | None):
    """Clamp CPU thread usage for torch and BLAS backends.

    Ensures a consistent ceiling across libraries and provides a single place to
    tighten thread counts when running multiple experiments in parallel.
    """
    thread_count = max(1, thread_count or 1)
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(max(1, thread_count // 2))
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[var] = str(thread_count)


def load_all(args: argparse.ArgumentParser):
    # thread_count = max(1, getattr(args, "torch_threads", 4))
    # apply_thread_limits(thread_count)
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    model = LoadModel(model_name=args.model, Chans=args.channel, Samples=int(args.fs*args.timepoint), n_classes=args.nclass).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initlr)

    return model, optimizer, device
