from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    parser = argparse.ArgumentParser(description="Feature visualization + entropy histogram")
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
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"))
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint path for the unlearned model to visualize.",
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


def entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


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


def subsample(result: ModelResult, max_points: int, seed: int) -> ModelResult:
    if len(result.features) <= max_points:
        return result
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(result.features), size=max_points, replace=False)
    return ModelResult(
        name=result.name,
        features=result.features[idx],
        labels=result.labels[idx],
        logits=result.logits[idx],
    )


def tsne_plot(results: List[ModelResult], perplexity: float, seed: int, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(12, 4))
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        tsne = TSNE(n_components=2, init="pca", random_state=seed, perplexity=perplexity)
        embedded = tsne.fit_transform(result.features)
        unique_labels = np.unique(result.labels)
        colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(unique_labels))))
        for idx, label in enumerate(unique_labels):
            mask = result.labels == label
            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                s=8,
                c=[colors[idx]],
                label=f"class {int(label)}",
            )
        ax.set_title(result.name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "feature_tsne.png"
    fig.savefig(outpath, dpi=200)
    return outpath


def entropy_plot(results: List[ModelResult], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for result in results:
        ent = entropy_from_logits(result.logits)
        ax.hist(np.log(ent), bins=20, alpha=0.6, label=result.name)
    ax.set_xlabel("Log of Entropy")
    ax.set_ylabel("Number")
    ax.set_title("Distribution of Output Entropy")
    ax.legend()
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "entropy_hist.png"
    fig.savefig(outpath, dpi=200)
    return outpath


def main() -> None:
    args = parse_args()
    args = set_args(args)
    apply_thread_limits(args.torch_threads)
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

    loaders = Load_MU_Dataloader(
        seed=args.seed,
        dataset=args.dataset,
        batchsize=args.bs,
        is_task=args.is_task,
        forget_subject=args.forget_subject,
    )

    # ckpt = args.ckpt
    ckpt = str(project_root / "ModelSave" / args.dataset / f"DiCE_{args.model}_{args.seed}_forget{args.forget_subject}.pth")
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

    results = [subsample(result, args.max_points, args.seed) for result in results]

    tsne_path = tsne_plot(results, args.perplexity, args.seed, args.outdir)
    entropy_path = entropy_plot(results, args.outdir)

    summary = {
        "tsne_plot": str(tsne_path),
        "entropy_plot": str(entropy_path),
        "forget_subject": loaders.get("forget_subject"),
        "compare_splits": args.compare_splits,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()