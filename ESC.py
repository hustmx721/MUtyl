import torch
from torch import nn
from types import MethodType
from tqdm import tqdm


def attach_esc_set(model: nn.Module) -> nn.Module:
    """Attach a minimal esc_set implementation to a model.

    This is a lightweight compatibility shim for models that do not define
    esc_set themselves. It registers or updates a projection matrix buffer
    named ``esc`` that can be consumed in the model's forward pass.

    Note: the model must still apply the projection (using ``model.esc``) in
    forward for ESC to have an effect. This helper only provides the setter.
    """

    def _esc_set(self, u, esc_t: bool = False):
        if esc_t:
            if hasattr(self, "esc"):
                self.esc = u
            else:
                self.register_buffer("esc", u.T)
        else:
            if hasattr(self, "esc"):
                self.esc = u @ self.esc
            else:
                self.register_buffer("esc", u)

    if not hasattr(model, "esc_set"):
        model.esc_set = MethodType(_esc_set, model)
    return model


class ESC:
    """Erasing Space Concept (ESC) unlearning implementation.

    This class mirrors the core algorithm in baselines/ESC/unlearn.py by:
    1) extracting pre-logits features from the forget set,
    2) running SVD on the feature matrix,
    3) applying a projection in feature space via model.esc_set.

    If use_esc_t is True, it follows the ESC-T procedure with a learnable
    mask over singular vectors.

    Notes:
        ESC requires the model to expose `esc_set(u, esc_t=False)`. This
        method is defined in baselines/ESC/models.py and baselines/ESC/
        vision_transformer.py, where it registers or updates a projection
        matrix used during forward to erase the target subspace. The check
        below ensures the provided model supports that projection hook.
    """

    def __init__(
        self,
        p: float = 1.5,
        threshold: float = 0.7,
        lr: float = 1e-3,
        epoch: int = 50,
        device: torch.device | None = None,
        use_esc_t: bool = False,
        verbose: bool = True,
    ) -> None:
        self.p = p
        self.threshold = threshold
        self.lr = lr
        self.epoch = epoch
        self.device = device
        self.use_esc_t = use_esc_t
        self.verbose = verbose

    def unlearn(self, model: nn.Module, forget_loader) -> nn.Module:
        """Apply ESC/ESC-T unlearning on the given model.

        Args:
            model: A model that exposes esc_set(u, esc_t=False) and returns
                a dict with key 'pre_logits' when called with all=True.
            forget_loader: DataLoader for the forget set.

        Returns:
            The unlearned model (modified in-place).
        """
        if not hasattr(model, "esc_set"):
            attach_esc_set(model)

        device = self.device or next(model.parameters()).device
        model.eval()

        feat_log = self._collect_features(model, forget_loader, device)
        u = self._svd(feat_log, device)

        if self.use_esc_t:
            self._apply_esc_t(model, forget_loader, u, device)
        else:
            self._apply_esc(model, u)

        return model

    def _collect_features(self, model: nn.Module, loader, device: torch.device) -> torch.Tensor:
        data_len = len(loader.dataset)
        with torch.no_grad():
            sample_x, _ = next(iter(loader))
            sample_x = sample_x.to(device, non_blocking=True)
            sample_out = model(sample_x, all=True)
            feat_dim = sample_out["pre_logits"].shape[1]

        feat_log = torch.zeros(data_len, feat_dim)

        with torch.no_grad():
            start = 0
            for x, _ in tqdm(loader, disable=not self.verbose, desc="ESC: feature extract"):
                x = x.to(device, non_blocking=True)
                output = model(x, all=True)
                batch_feats = output["pre_logits"].detach().cpu()
                end = start + batch_feats.shape[0]
                feat_log[start:end, :] = batch_feats
                start = end

        return feat_log

    def _svd(self, feat_log: torch.Tensor, device: torch.device) -> torch.Tensor:
        u, _, _ = torch.linalg.svd(feat_log.T.to(device), full_matrices=False)
        return u

    def _apply_esc(self, model: nn.Module, u: torch.Tensor) -> None:
        prune_k = int(u.shape[1] * self.p / 100)
        u_p = u[:, prune_k:]
        model.esc_set(u_p)

    def _apply_esc_t(
        self,
        model: nn.Module,
        loader,
        u: torch.Tensor,
        device: torch.device,
    ) -> None:
        mask = torch.ones_like(u)
        criterion = nn.CrossEntropyLoss()

        for _ in tqdm(range(self.epoch), disable=not self.verbose, desc="ESC-T: optimize"):
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                mask = mask.detach()
                mask.requires_grad_(True)

                model.esc_set(u * mask, esc_t=True)
                _, logits = model(x)

                pred = logits.argmax(dim=1)
                learned = y == pred

                if learned.any():
                    loss = -criterion(logits[learned], y[learned])
                    loss.backward()

                    if mask.grad is not None:
                        with torch.no_grad():
                            mask = mask - self.lr * mask.grad
                            mask = torch.clamp(mask, min=0, max=1)
                    mask.grad = None

            model.esc_set(u * mask, esc_t=True)

            model.eval()
            with torch.no_grad():
                num_hits = 0
                for x, y in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    _, logits = model(x)
                    pred = logits.argmax(dim=1)
                    num_hits += (y == pred).sum().item()
            if num_hits == 0:
                break

        mask = (mask > self.threshold).to(mask.dtype)
        model.esc_set(u * mask, esc_t=True)
