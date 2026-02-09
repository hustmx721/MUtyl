from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from utils.MULoader import Load_MU_Dataloader
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, set_args
from utils.models import LoadModel

project_root = Path(__file__).resolve().parent.parent

@dataclass
class ModelResult:
    name: str
    features: np.ndarray
    labels: np.ndarray
    logits: np.ndarray


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature extraction for visualization")
    parser.add_argument("--dataset", type=str, default="004")
    parser.add_argument("--model", type=str, default="EEGNet")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--is_task", type=str2bool, default=True)
    parser.add_argument("--forget_subject", type=int, default=None)
    parser.add_argument(
        "--compare_splits",
        action="store_true",
        help="Compare forget vs remain splits for the same model checkpoint.",
    )
    parser.add_argument("--features_dir", type=Path, default="/mnt/data1/tyl/MachineUnlearning/MUtyl/features")
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint path for the unlearned model.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="unlearned model",
        help="Display name for the checkpoint.",
    )
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


def collect_features(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    logits_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                features, logits = output[0], output[1]
            else:
                logits = output
                features = logits
            features_list.append(features.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
    return (
        np.concatenate(features_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(logits_list, axis=0),
    )


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.ckpt is not None:
        return args.ckpt
    return project_root / "MUtyl" / "ModelSave" / args.dataset / f"DiCE_{args.model}_{args.seed}_forget{args.forget_subject}.pth"


def save_feature_bundle(result: ModelResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "name": result.name,
            "features": result.features,
            "labels": result.labels,
            "logits": result.logits,
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    args = set_args(args)
    apply_thread_limits(args.torch_threads)
    set_seed(args.seed)
    subjects = np.arange(0, 9)

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

    for subject in subjects:
        args.forget_subject = int(subject)

        loaders = Load_MU_Dataloader(
            seed=args.seed,
            dataset=args.dataset,
            batchsize=args.bs,
            is_task=args.is_task,
            forget_subject=args.forget_subject,
        )

        ckpt = resolve_checkpoint(args)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        model = LoadModel(
            model_name=args.model,
            Chans=args.channel,
            Samples=int(args.fs * args.timepoint),
            n_classes=args.nclass,
        ).to(device)
        load_checkpoint(model, ckpt, device)

        results: List[ModelResult] = []
        split_keys = ["test_loader_forget", "test_loader_remain"] if args.compare_splits else ["test_loader_forget"]
        split_names = ["forget", "remain"] if args.compare_splits else ["forget"]
        for split_key, split_name in zip(split_keys, split_names):
            data_loader = loaders[split_key]
            features, labels, logits = collect_features(model, data_loader, device)
            results.append(
                ModelResult(
                    name=f"{args.name} ({split_name})",
                    features=features,
                    labels=labels,
                    logits=logits,
                )
            )
            output_name = f"{args.dataset}_{args.model}_seed{args.seed}_forget{args.forget_subject}_{split_name}.pt"
            output_path = args.features_dir / output_name
            save_feature_bundle(results[-1], output_path)
            print(f"Saved features to: {output_path}")


if __name__ == "__main__":
    main()
