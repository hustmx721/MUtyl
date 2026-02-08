from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


@dataclass
class ModelResult:
    name: str
    features: np.ndarray
    labels: np.ndarray
    logits: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize saved feature tensors")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to saved .pt feature bundles.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def load_bundle(path: Path) -> ModelResult:
    payload = torch.load(path, map_location="cpu")
    return ModelResult(
        name=payload.get("name", path.stem),
        features=payload["features"],
        labels=payload["labels"],
        logits=payload["logits"],
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
    results = [load_bundle(path) for path in args.inputs]
    results = [subsample(result, args.max_points, args.seed) for result in results]
    tsne_path = tsne_plot(results, args.perplexity, args.seed, args.outdir)
    entropy_path = entropy_plot(results, args.outdir)
    print(f"Saved t-SNE plot to: {tsne_path}")
    print(f"Saved entropy plot to: {entropy_path}")


if __name__ == "__main__":
    main()
