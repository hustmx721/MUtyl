import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import nn


@dataclass
class LAFConfig:
    """Configuration for Label-Agnostic Forgetting (LAF) unlearning."""

    vae_epochs: int = 10
    vae_lr: float = 1e-3
    vae_kld_weight: float = 0.025
    unlearning_epochs: int = 10
    unlearning_lr_ue: float = 1e-4
    unlearning_lr_ra: float = 1e-4
    contrastive_temp: float = 1.0
    classifier_epochs: int = 5
    classifier_lr: float = 1e-4
    device: torch.device | None = None
    verbose: bool = True


class LAF:
    """Label-Agnostic Forgetting (LAF) unlearning implementation.

    This mirrors the core algorithm in baselines/LAF by:
    1) training variational autoencoders on remaining and forget data,
    2) updating the feature extractor with reconstruction + contrastive losses,
    3) refining the classifier head with remaining data.

    The model is expected to return (logits, embedding) in its forward pass.
    """

    def __init__(self, config: LAFConfig | None = None) -> None:
        self.config = config or LAFConfig()

    def unlearn(
        self,
        model: nn.Module,
        loaders: Dict[str, Iterable],
        s_vae: nn.Module,
        u_vae: nn.Module,
        train_vae: bool = True,
    ) -> Tuple[nn.Module, Dict[str, list]]:
        """Run the LAF pipeline and return the updated model.

        Args:
            model: The target model to be unlearned.
            loaders: A dict with keys ``unlearn`` and ``remain`` (or ``train``).
            s_vae: VAE trained on remaining data representations.
            u_vae: VAE trained on unlearn data representations.
            train_vae: Whether to (re)train the VAEs before unlearning.

        Returns:
            The updated model and a dict of extractor losses.
        """
        device = self.config.device or next(model.parameters()).device
        model = model.to(device)

        remain_loader = self._select_loader(loaders, preferred="remain", fallback="train")
        if train_vae:
            s_vae = self._train_vae(model, s_vae, remain_loader, device)
            u_vae = self._train_vae(model, u_vae, loaders["unlearn"], device)

        model, loss1_list, loss2_list = self._extractor_unlearning(
            model,
            s_vae,
            u_vae,
            loaders,
            device,
        )

        model = self._classifier_unlearning(model, loaders, device)

        return model, {"extractor_nmse": loss1_list, "extractor_contrastive": loss2_list}

    def _select_loader(self, loaders: Dict[str, Iterable], preferred: str, fallback: str) -> Iterable:
        if preferred in loaders:
            return loaders[preferred]
        if fallback in loaders:
            return loaders[fallback]
        raise KeyError(f"Expected loaders to include '{preferred}' or '{fallback}'.")

    def _vae_loss_function(self, x_out: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.mse_loss(x_out, x, reduction="sum") / x.shape[0]
        kld = -0.5 * torch.sum(torch.log(sigma.pow(2) + 1e-8) + 1 - mu.pow(2) - sigma.pow(2)) / x.shape[0]
        return bce + kld * self.config.vae_kld_weight

    def _train_vae(
        self,
        trained_model: nn.Module,
        vae: nn.Module,
        loader: Iterable,
        device: torch.device,
    ) -> nn.Module:
        trained_model.eval()
        vae = vae.to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=self.config.vae_lr)

        for epoch in range(self.config.vae_epochs):
            all_loss = 0.0
            count = 0
            vae.train()

            for x, _ in loader:
                x = x.to(device, non_blocking=True)

                _, embedding = trained_model(x)
                optimizer.zero_grad()
                e_out, _, mu, sigma = vae(embedding)
                loss = self._vae_loss_function(e_out, embedding, mu, sigma)
                loss.backward()
                all_loss += loss.item()
                count += 1
                optimizer.step()

            if self.config.verbose:
                epoch_loss = all_loss / max(count, 1)
                print(f"LAF VAE Training Epoch: {epoch} Loss: {epoch_loss}")

        return vae

    def _extractor_loss_nmse(self, e_out: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        mse = nn.MSELoss(reduction="none")(e_out, e)
        mse = mse.sum(dim=1)
        return torch.sum(torch.exp(mse / (mse + 1)))

    def _extractor_loss_cosine(
        self,
        s_e: torch.Tensor,
        s_e_teacher: torch.Tensor,
        u_e: torch.Tensor,
        u_e_teacher: torch.Tensor,
        temp: float,
    ) -> torch.Tensor:
        loss_fn = nn.CosineEmbeddingLoss(reduction="none")
        s_targets = torch.ones(s_e.shape[0], device=s_e.device)
        u_targets = torch.ones(u_e.shape[0], device=u_e.device)
        cos1 = loss_fn(s_e, s_e_teacher, s_targets)
        cos2 = loss_fn(u_e, u_e_teacher, u_targets)
        return torch.sum(torch.log(torch.exp(cos1) / torch.sum(torch.exp(cos2 / temp))))

    def _extractor_unlearning(
        self,
        trained_model: nn.Module,
        s_vae: nn.Module,
        u_vae: nn.Module,
        loaders: Dict[str, Iterable],
        device: torch.device,
    ) -> Tuple[nn.Module, list, list]:
        trained_model = trained_model.to(device)
        s_vae = s_vae.to(device)
        s_vae.eval()
        u_vae = u_vae.to(device)
        u_vae.eval()
        teacher_model = copy.deepcopy(trained_model).to(device)
        teacher_model.eval()

        optimizer1 = torch.optim.Adam(trained_model.parameters(), lr=self.config.unlearning_lr_ue)
        optimizer2 = torch.optim.Adam(trained_model.parameters(), lr=self.config.unlearning_lr_ra)
        all_loss_list = []
        all_loss2_list = []

        remain_loader = self._select_loader(loaders, preferred="remain", fallback="train")

        for epoch in range(self.config.unlearning_epochs):
            all_loss = 0.0
            all_loss2 = 0.0
            count = 0

            for (u_data, _), (s_data, _) in zip(loaders["unlearn"], remain_loader):
                u_data = u_data.to(device, non_blocking=True)
                s_data = s_data.to(device, non_blocking=True)

                _, s_e = trained_model(s_data)
                s_e_out, _, _, _ = s_vae(s_e)

                _, u_e = trained_model(u_data)
                u_u_e_out, _, _, _ = u_vae(u_e)

                loss2 = self._extractor_loss_nmse(s_e_out, s_e) - self._extractor_loss_nmse(u_u_e_out, u_e)

                optimizer1.zero_grad()
                loss2.backward()
                optimizer1.step()

                _, s_e = trained_model(s_data)
                _, s_e_teacher = teacher_model(s_data)
                _, u_e = trained_model(u_data)
                _, u_e_teacher = teacher_model(u_data)

                loss3 = self._extractor_loss_cosine(
                    s_e,
                    s_e_teacher,
                    u_e,
                    u_e_teacher,
                    self.config.contrastive_temp,
                )
                optimizer2.zero_grad()
                loss3.backward()
                optimizer2.step()

                all_loss += loss2.item()
                all_loss2 += loss3.item()
                count += 1

            all_loss_list.append(all_loss / max(count, 1))
            all_loss2_list.append(all_loss2 / max(count, 1))
            if self.config.verbose:
                print(all_loss_list[-1])
                print(all_loss2_list[-1])

        return trained_model, all_loss_list, all_loss2_list

    def _classifier_unlearning(
        self,
        trained_model: nn.Module,
        loaders: Dict[str, Iterable],
        device: torch.device,
    ) -> nn.Module:
        trained_model = trained_model.to(device)
        loss_func = nn.CrossEntropyLoss()

        if hasattr(trained_model, "fc2"):
            params = trained_model.fc2.parameters()
        else:
            params = trained_model.parameters()

        optimizer = torch.optim.Adam(params, lr=self.config.classifier_lr)
        remain_loader = self._select_loader(loaders, preferred="remain", fallback="train")

        for epoch in range(self.config.classifier_epochs):
            all_loss = 0.0
            count = 0

            for (_, _), (x, y) in zip(loaders["unlearn"], remain_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_out, _ = trained_model(x)
                loss = loss_func(y_out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_loss += loss.item()
                count += 1

            if self.config.verbose:
                epoch_loss = all_loss / max(count, 1)
                print(f"LAF classifier Epoch: {epoch} Loss: {epoch_loss}")

        return trained_model
