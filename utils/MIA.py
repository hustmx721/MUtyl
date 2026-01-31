"""Membership inference attack utilities.

This module provides a lightweight black-box membership inference attack that
works with the project's DataLoader/model conventions. It supports the common
"shadow/target" split and outputs attack accuracies for several classical
metrics (confidence, entropy, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def _unpack_logits(output):
    if isinstance(output, (tuple, list)):
        if len(output) != 2:
            raise ValueError("Expected model to return (features, logits).")
        _, logits = output
        return logits
    return output


def _infer_device(model: torch.nn.Module, device=None):
    if device is not None:
        return device
    return next(model.parameters()).device


@dataclass
class MIASummary:
    attack_acc: float
    train_acc: float
    test_acc: float


class BlackBoxMIA:
    def __init__(
        self,
        shadow_train_performance,
        shadow_test_performance,
        target_train_performance,
        target_test_performance,
        num_classes: int,
    ):
        """Black-box membership inference benchmark.

        Each input contains model predictions (num_samples, num_classes) and
        ground-truth labels (num_samples,).
        """

        self.num_classes = num_classes

        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance

        self.s_tr_corr = (
            np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels
        ).astype(int)
        self.s_te_corr = (
            np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels
        ).astype(int)
        self.t_tr_corr = (
            np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels
        ).astype(int)
        self.t_te_corr = (
            np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels
        ).astype(int)

        self.s_tr_conf = np.take_along_axis(
            self.s_tr_outputs, self.s_tr_labels[:, None], axis=1
        )
        self.s_te_conf = np.take_along_axis(
            self.s_te_outputs, self.s_te_labels[:, None], axis=1
        )
        self.t_tr_conf = np.take_along_axis(
            self.t_tr_outputs, self.t_tr_labels[:, None], axis=1
        )
        self.t_te_conf = np.take_along_axis(
            self.t_te_outputs, self.t_te_labels[:, None], axis=1
        )

        self.s_tr_entr = self._entropy(self.s_tr_outputs)
        self.s_te_entr = self._entropy(self.s_te_outputs)
        self.t_tr_entr = self._entropy(self.t_tr_outputs)
        self.t_te_entr = self._entropy(self.t_te_outputs)

        self.s_tr_m_entr = self._modified_entropy(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._modified_entropy(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._modified_entropy(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._modified_entropy(self.t_te_outputs, self.t_te_labels)

    @staticmethod
    def _log_value(probs, eps=1e-30):
        return -np.log(np.maximum(probs, eps))

    def _entropy(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _modified_entropy(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[
            range(true_labels.size), true_labels
        ]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[
            range(true_labels.size), true_labels
        ]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    @staticmethod
    def _threshold_from_shadow(shadow_train, shadow_test):
        value_list = np.concatenate((shadow_train, shadow_test))
        best_thre, best_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(shadow_train >= value) / (len(shadow_train) + 0.0)
            te_ratio = np.sum(shadow_test < value) / (len(shadow_test) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > best_acc:
                best_thre, best_acc = value, acc
        return best_thre

    def _mia_via_correctness(self) -> MIASummary:
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        t_te_acc = 1 - np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        return MIASummary(
            attack_acc=0.5 * (t_tr_acc + t_te_acc),
            train_acc=t_tr_acc,
            test_acc=t_te_acc,
        )

    def _mia_via_threshold(self, s_tr_values, s_te_values, t_tr_values, t_te_values) -> MIASummary:
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            shadow_train = s_tr_values[self.s_tr_labels == num]
            shadow_test = s_te_values[self.s_te_labels == num]
            if shadow_train.size == 0 or shadow_test.size == 0:
                continue
            threshold = self._threshold_from_shadow(shadow_train, shadow_test)
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= threshold)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < threshold)
        t_tr_acc = t_tr_mem / (len(self.t_tr_labels) + 0.0)
        t_te_acc = t_te_non_mem / (len(self.t_te_labels) + 0.0)
        return MIASummary(
            attack_acc=0.5 * (t_tr_acc + t_te_acc),
            train_acc=t_tr_acc,
            test_acc=t_te_acc,
        )

    def run(self, methods: Iterable[str] | None = None):
        if methods is None:
            methods = ["correctness", "confidence", "entropy", "modified_entropy"]
        results = {}
        for method in methods:
            if method == "correctness":
                results[method] = self._mia_via_correctness()
            elif method == "confidence":
                results[method] = self._mia_via_threshold(
                    self.s_tr_conf,
                    self.s_te_conf,
                    self.t_tr_conf,
                    self.t_te_conf,
                )
            elif method == "entropy":
                results[method] = self._mia_via_threshold(
                    -self.s_tr_entr,
                    -self.s_te_entr,
                    -self.t_tr_entr,
                    -self.t_te_entr,
                )
            elif method == "modified_entropy":
                results[method] = self._mia_via_threshold(
                    -self.s_tr_m_entr,
                    -self.s_te_m_entr,
                    -self.t_tr_m_entr,
                    -self.t_te_m_entr,
                )
            else:
                raise ValueError(f"Unsupported MIA method: {method}")
        return results


def collect_performance(data_loader, model, device=None):
    probs = []
    labels = []
    model.eval()
    device = _infer_device(model, device)

    for batch in data_loader:
        if len(batch) == 3:
            data, target, _ = batch
        else:
            data, target = batch
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = _unpack_logits(model(data))
            prob = F.softmax(output, dim=-1)

        probs.append(prob)
        labels.append(target)

    if not probs:
        raise ValueError("Empty dataloader provided to collect_performance.")

    return torch.cat(probs).cpu().numpy(), torch.cat(labels).cpu().numpy()


def run_mia(
    shadow_train_loader,
    shadow_test_loader,
    target_train_loader,
    target_test_loader,
    model,
    device=None,
    methods: Iterable[str] | None = None,
):
    """Run membership inference attack with a shadow/target split.

    Args:
        shadow_train_loader: Shadow model train data (member).
        shadow_test_loader: Shadow model test data (non-member).
        target_train_loader: Target model train data (member).
        target_test_loader: Target model test data (non-member).
        model: Model under evaluation.
        device: Torch device (optional).
        methods: List of attack methods.
    """

    device = _infer_device(model, device)
    shadow_train_perf = collect_performance(shadow_train_loader, model, device)
    shadow_test_perf = collect_performance(shadow_test_loader, model, device)
    target_train_perf = collect_performance(target_train_loader, model, device)
    target_test_perf = collect_performance(target_test_loader, model, device)

    num_classes = shadow_train_perf[0].shape[1]
    mia = BlackBoxMIA(
        shadow_train_perf,
        shadow_test_perf,
        target_train_perf,
        target_test_perf,
        num_classes=num_classes,
    )
    return mia.run(methods=methods)


def run_mia_from_mu_loaders(loaders, model, device=None, methods: Iterable[str] | None = None):
    """Convenience wrapper for MU-style loaders.

    Expected keys in ``loaders``:
        - remain_train_loader
        - test_loader_remain
        - forget_train_loader
        - test_loader_forget
    """

    required = {
        "remain_train_loader",
        "test_loader_remain",
        "forget_train_loader",
        "test_loader_forget",
    }
    missing = required - set(loaders.keys())
    if missing:
        raise KeyError(f"Missing required MU loader keys: {sorted(missing)}")

    return run_mia(
        loaders["remain_train_loader"],
        loaders["test_loader_remain"],
        loaders["forget_train_loader"],
        loaders["test_loader_forget"],
        model,
        device=device,
        methods=methods,
    )
