from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from tqdm import tqdm


@dataclass
class GAConfig:
    """Configuration for Gradient Ascent (GA) unlearning."""

    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    retain_weight: float = 0.0
    normalize_loss: bool = True
    label_index: int = -1
    device: torch.device | None = None
    verbose: bool = True


class GA:
    """Gradient Ascent (GA) unlearning implementation.

    This baseline performs gradient ascent on the forget set loss. It mirrors
    the update: theta' = theta* + eta * sum_{(x,y) in F} grad L(f_theta(x), y).
    When a retain loader is provided, it optionally adds a retain loss term for
    stability (retain_weight = 0 keeps the pure GA update).

    The model is expected to output logits (or a tuple where the second
    element is logits).
    """

    def __init__(self, config: GAConfig | None = None) -> None:
        self.config = config or GAConfig()

    def unlearn(self, model: nn.Module, loaders: Dict[str, Iterable]) -> nn.Module:
        """Apply GA unlearning.

        Args:
            model: The model to be unlearned.
            loaders: Dict containing ``forget`` and optionally ``retain`` or ``train`` loaders.

        Returns:
            The unlearned model.
        """
        device = self.config.device or next(model.parameters()).device
        model = model.to(device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(reduction="sum")

        forget_loader = loaders["forget"]
        retain_loader = self._select_loader(loaders, preferred="retain", fallback="train", required=False)

        for _ in range(self.config.epochs):
            iterator = tqdm(forget_loader, disable=not self.config.verbose, desc="GA: forget")
            retain_iter = iter(retain_loader) if retain_loader is not None else None

            for batch in iterator:
                inputs, labels = self._split_batch(batch)
                inputs = self._move_to_device(inputs, device)
                labels = labels.to(device, non_blocking=True)

                logits = self._forward(model, inputs)
                loss_forget = criterion(logits, labels)
                if self.config.normalize_loss:
                    loss_forget = loss_forget / max(labels.shape[0], 1)
                loss = -loss_forget

                if retain_iter is not None and self.config.retain_weight > 0:
                    try:
                        retain_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        retain_batch = next(retain_iter)
                    retain_inputs, retain_labels = self._split_batch(retain_batch)
                    retain_inputs = self._move_to_device(retain_inputs, device)
                    retain_labels = retain_labels.to(device, non_blocking=True)
                    retain_logits = self._forward(model, retain_inputs)
                    retain_loss = criterion(retain_logits, retain_labels)
                    if self.config.normalize_loss:
                        retain_loss = retain_loss / max(retain_labels.shape[0], 1)
                    loss = loss + self.config.retain_weight * retain_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

    def _select_loader(
        self,
        loaders: Dict[str, Iterable],
        preferred: str,
        fallback: str,
        required: bool = True,
    ) -> Iterable | None:
        if preferred in loaders:
            return loaders[preferred]
        if fallback in loaders:
            return loaders[fallback]
        if required:
            raise KeyError(f"Expected loaders to include '{preferred}' or '{fallback}'.")
        return None

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
            output = model(*inputs)
        else:
            output = model(inputs)
        if isinstance(output, (tuple, list)) and len(output) > 1:
            return output[1]
        return output
