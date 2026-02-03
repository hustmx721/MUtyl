import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class DistillKL(nn.Module):
    """Distilling the knowledge in a neural network (KL with temperature)."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
        p_s = F.log_softmax(logits_s / self.temperature, dim=1)
        p_t = F.softmax(logits_t / self.temperature, dim=1)
        return F.kl_div(p_s, p_t, reduction="sum") * (self.temperature**2) / logits_s.shape[0]


@dataclass
class SCRUBConfig:
    """Configuration for SCRUB unlearning."""

    epochs: int = 10
    m_steps: int = 1
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_decay_rate: float = 0.1
    lr_decay_epochs: Sequence[int] = field(default_factory=lambda: [3, 5, 9])
    temperature: float = 4.0
    scrub_gamma: float = 0.99
    scrub_alpha: float = 0.001
    smoothing: float = 0.0
    label_index: int = -1
    device: torch.device | None = None
    verbose: bool = True


class SCRUB:
    """SCRUB unlearning implementation.

    This mirrors the SCRUB procedure from baselines/Unlearn-WorstCase/data-wise/unlearn/SCRUB.py:
    1) maximize the KL divergence on the forget set for the first m_steps epochs,
    2) then minimize classification + KL divergence on the retain set,
    3) optionally regularize toward the initial weights via a smoothing term.
    """

    def __init__(self, config: SCRUBConfig | None = None) -> None:
        self.config = config or SCRUBConfig()

    def unlearn(
        self,
        model: nn.Module,
        loaders: Dict[str, Iterable],
        teacher_model: nn.Module | None = None,
    ) -> nn.Module:
        """Apply SCRUB unlearning on the given model.

        Args:
            model: The model to be unlearned (updated in-place).
            loaders: Dict containing ``forget`` and ``retain`` (or ``train``) loaders.
            teacher_model: Optional frozen teacher; defaults to a copy of ``model``.

        Returns:
            The unlearned model.
        """
        device = self.config.device or next(model.parameters()).device
        model = model.to(device)
        model.train()

        teacher = teacher_model or copy.deepcopy(model)
        teacher = teacher.to(device)
        teacher.eval()

        swa_model = copy.deepcopy(model).to(device) if self.config.smoothing > 0 else None

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(self.config.temperature)

        retain_loader = self._select_loader(loaders, preferred="retain", fallback="train")
        forget_loader = loaders["forget"]

        for epoch in range(self.config.epochs):
            self._adjust_learning_rate(optimizer, epoch)

            if epoch < self.config.m_steps:
                iterator = tqdm(forget_loader, disable=not self.config.verbose, desc="SCRUB: forget")
                for batch in iterator:
                    inputs, labels = self._split_batch(batch)
                    inputs = self._move_to_device(inputs, device)
                    labels = labels.to(device, non_blocking=True).long()

                    logits_s = self._forward(model, inputs)
                    with torch.no_grad():
                        logits_t = self._forward(teacher, inputs)

                    loss_div = criterion_div(logits_s, logits_t)
                    loss = -loss_div + self._param_dist(model, swa_model)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                iterator = tqdm(retain_loader, disable=not self.config.verbose, desc="SCRUB: retain")
                for batch in iterator:
                    inputs, labels = self._split_batch(batch)
                    inputs = self._move_to_device(inputs, device)
                    labels = labels.to(device, non_blocking=True).long()

                    logits_s = self._forward(model, inputs)
                    with torch.no_grad():
                        logits_t = self._forward(teacher, inputs)

                    loss_cls = criterion_cls(logits_s, labels)
                    loss_div = criterion_div(logits_s, logits_t)
                    loss = self.config.scrub_gamma * loss_cls + self.config.scrub_alpha * loss_div
                    loss = loss + self._param_dist(model, swa_model)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        return model

    def _select_loader(self, loaders: Dict[str, Iterable], preferred: str, fallback: str) -> Iterable:
        if preferred in loaders:
            return loaders[preferred]
        if fallback in loaders:
            return loaders[fallback]
        raise KeyError(f"Expected loaders to include '{preferred}' or '{fallback}'.")

    def _adjust_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        steps = np.sum(epoch > np.asarray(self.config.lr_decay_epochs))
        new_lr = self.config.lr
        if steps > 0:
            new_lr = self.config.lr * (self.config.lr_decay_rate**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def _param_dist(self, model: nn.Module, swa_model: nn.Module | None) -> torch.Tensor:
        if swa_model is None or self.config.smoothing <= 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        dist = 0.0
        for p1, p2 in zip(model.parameters(), swa_model.parameters()):
            dist += torch.norm(p1 - p2, p="fro")
        return self.config.smoothing * dist

    def _split_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[object, torch.Tensor]:
        if not isinstance(batch, (tuple, list)):
            raise ValueError("Expected batch to be a tuple or list of tensors.")
        label_index = self.config.label_index
        if label_index < 0:
            label_index = len(batch) + label_index
        if label_index < 0 or label_index >= len(batch):
            raise ValueError("label_index is out of range for the provided batch.")

        labels = batch[label_index]
        inputs: List[torch.Tensor] = []
        for idx, item in enumerate(batch):
            if idx == label_index:
                continue
            inputs.append(item)

        if len(inputs) == 1:
            return inputs[0], labels
        return inputs, labels

    def _move_to_device(self, inputs: object, device: torch.device) -> object:
        if isinstance(inputs, (list, tuple)):
            return [self._move_to_device(x, device) for x in inputs]
        if torch.is_tensor(inputs):
            return inputs.to(device, non_blocking=True)
        return inputs

    def _forward(self, model: nn.Module, inputs: object) -> torch.Tensor:
        if isinstance(inputs, (list, tuple)):
            return model(*inputs)
        return model(inputs)
