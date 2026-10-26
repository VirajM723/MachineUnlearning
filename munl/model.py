"""Network architecture and a deterministic training loop."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressor(nn.Module):
    """Fully connected regressor: in -> 512 -> 512 -> 64 -> 1, ReLU + dropout."""

    def __init__(self, in_features: int, hidden: tuple[int, ...] = (512, 512, 64), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for i, width in enumerate(hidden):
            layers += [nn.Linear(prev, width), nn.ReLU()]
            # No dropout right before the output layer, matching the original design.
            if i < len(hidden) - 1:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_epochs(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    batch_size: int,
    seed: int,
) -> float:
    """Train ``model`` in place and return the mean loss of the final epoch.

    All randomness (batch order, dropout masks) is drawn from ``seed``, so the
    same inputs always produce the same weights. SISA relies on this: replaying
    a slice after an unlearning request must be indistinguishable from having
    trained without the deleted point in the first place.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True, generator=generator)
    criterion = nn.MSELoss()
    device = next(model.parameters()).device

    model.train()
    epoch_loss = float("nan")
    for _ in range(epochs):
        total, count = 0.0, 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_X)
            count += len(batch_X)
        epoch_loss = total / max(count, 1)
    return epoch_loss


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Inference in eval mode (dropout off), returned on the CPU."""
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(X[i : i + batch_size].to(device)).cpu() for i in range(0, len(X), batch_size)])
