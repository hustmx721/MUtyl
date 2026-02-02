from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import torch
from torch import nn
from tqdm import tqdm


@dataclass
class SISAConfig:
    """Configuration for SISA unlearning."""

    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    label_index: int = -1
    device: torch.device | None = None
    verbose: bool = True


class SISA:
    """SISA (Sharded, Isolated, Sliced, Aggregated) unlearning baseline.

    This implementation trains one model per shard and supports re-training
    only the shards that are affected by the forget set. Predictions are
    aggregated by averaging logits from each shard model.
    """

    def __init__(self, config: SISAConfig | None = None) -> None:
        self.config = config or SISAConfig()

    def train_shards(
        self,
        model_factory: Callable[[], nn.Module],
        shard_loaders: List[Iterable],
    ) -> List[nn.Module]:
        """Train one model per shard and return the list of shard models."""
        models: List[nn.Module] = []
        for shard_id, loader in enumerate(shard_loaders):
            model = model_factory()
            model = self._train_model(model, loader, shard_id)
            models.append(model)
        return models

    def unlearn(
        self,
        shard_models: List[nn.Module],
        model_factory: Callable[[], nn.Module],
        shard_loaders: List[Iterable],
        forget_shards: List[int],
    ) -> List[nn.Module]:
        """Re-train only the affected shards and return updated models."""
        updated_models = list(shard_models)
        for shard_id in forget_shards:
            if shard_id < 0 or shard_id >= len(shard_loaders):
                raise ValueError(f"shard_id {shard_id} is out of range for shard_loaders.")
            model = model_factory()
            model = self._train_model(model, shard_loaders[shard_id], shard_id)
            updated_models[shard_id] = model
        return updated_models

    def predict(self, shard_models: List[nn.Module], inputs: object) -> torch.Tensor:
        """Aggregate logits from all shard models by averaging."""
        if not shard_models:
            raise ValueError("shard_models must be a non-empty list.")
        device = self.config.device or next(shard_models[0].parameters()).device
        logits_list = []
        for model in shard_models:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                logits = self._forward(model, self._move_to_device(inputs, device))
                logits_list.append(logits)
        return torch.stack(logits_list, dim=0).mean(dim=0)

    def _train_model(self, model: nn.Module, loader: Iterable, shard_id: int) -> nn.Module:
        device = self.config.device or next(model.parameters()).device
        model = model.to(device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.config.epochs):
            iterator = tqdm(
                loader,
                disable=not self.config.verbose,
                desc=f"SISA: train shard {shard_id}",
            )
            for batch in iterator:
                inputs, labels = self._split_batch(batch)
                inputs = self._move_to_device(inputs, device)
                labels = labels.to(device, non_blocking=True)

                logits = self._forward(model, inputs)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

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
