import copy

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class DELETE:
    """Decoupled Distillation to Erase (DELETE) unlearning implementation.

    This class mirrors the core algorithm in baselines/DELETE/method/delete.py:
    it performs KL distillation against a frozen copy of the original model,
    while suppressing the ground-truth class logits to encourage forgetting.
    """

    def __init__(
        self,
        epoch: int = 10,
        lr: float = 0.01,
        soft_label: str = "inf",
        device: torch.device | None = None,
        disable_bn: bool = False,
        verbose: bool = True,
    ) -> None:
        self.epoch = epoch
        self.lr = lr
        self.soft_label = soft_label
        self.device = device
        self.disable_bn = disable_bn
        self.verbose = verbose

    def unlearn(self, model: nn.Module, forget_loader) -> nn.Module:
        """Apply DELETE unlearning on the given model.

        Args:
            model: Model to be unlearned (updated in-place).
            forget_loader: DataLoader for the forget set.

        Returns:
            The unlearned model.
        """
        device = self.device or next(model.parameters()).device
        model.to(device)
        model.train()

        teacher = copy.deepcopy(model).to(device)
        teacher.eval()

        criterion = nn.KLDivLoss(reduction="batchmean")
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        for _ in tqdm(range(self.epoch), disable=not self.verbose, desc="DELETE: unlearn"):
            for x, y in forget_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if self.disable_bn:
                    for module in model.modules():
                        if isinstance(module, nn.BatchNorm2d):
                            module.eval()

                optimizer.zero_grad()

                with torch.no_grad():
                    _, pred_label = teacher(x)
                    if self.soft_label == "inf":
                        pred_label[torch.arange(x.shape[0]), y] = -1e10
                    else:
                        raise ValueError("Unknown soft label method")

                _, ori_logits = model(x)
                ori_logits = F.log_softmax(ori_logits, dim=1)
                pred_label = F.softmax(pred_label, dim=1)
                loss = criterion(ori_logits, pred_label)
                loss.backward()
                optimizer.step()

        return model
