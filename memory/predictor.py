"""
predictor.py — Action predictor with MoE awareness for Qwen 3.5.

Predicts the next latent state ĥ given current state h and tool embedding z,
with optional expert routing information.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ActionPredictor(nn.Module):
    """
    MLP that predicts ĥ_after from h_before and z_tool.

    Supports expert routing information for MoE models like Qwen 3.5.
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int = 128,
        num_experts: Optional[int] = None,
        hidden_sizes: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 512]

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        # Input dimension depends on whether we have expert info
        expert_dim = num_experts if num_experts else 0
        self.input_dim = hidden_dim + embed_dim + expert_dim

        # Build MLP
        layers = []
        prev_dim = self.input_dim
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, size),
                nn.LayerNorm(size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = size
        layers.append(nn.Linear(prev_dim, hidden_dim))

        self.net = nn.Sequential(*layers)

        # Optional expert predictor
        if num_experts:
            self.expert_predictor = nn.Linear(hidden_dim, num_experts)

        self.to(self._get_device())

    def _get_device(self):
        return next(self.parameters()).device if self._parameters else torch.device("cpu")

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        expert_routing: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h: (batch, hidden_dim) or (hidden_dim,)
            z: (batch, embed_dim) or (embed_dim,)
            expert_routing: (batch, num_experts) or None

        Returns:
            h_pred: predicted next state
            expert_pred: predicted expert distribution (if num_experts)
        """
        # Handle single samples
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if expert_routing is not None and expert_routing.dim() == 1:
            expert_routing = expert_routing.unsqueeze(0)

        # Ensure same batch size
        if h.size(0) != z.size(0):
            if h.size(0) == 1:
                h = h.expand(z.size(0), -1)
            elif z.size(0) == 1:
                z = z.expand(h.size(0), -1)
            else:
                raise ValueError(f"Batch size mismatch: h={h.size(0)}, z={z.size(0)}")

        # Concatenate inputs
        inputs = [h, z]
        if expert_routing is not None:
            inputs.append(expert_routing.float())

        x = torch.cat(inputs, dim=-1)
        h_pred = self.net(x)

        # Predict expert distribution if requested
        expert_pred = None
        if hasattr(self, 'expert_predictor'):
            expert_pred = self.expert_predictor(h_pred)

        if h_pred.size(0) == 1:
            h_pred = h_pred.squeeze(0)
            if expert_pred is not None and expert_pred.size(0) == 1:
                expert_pred = expert_pred.squeeze(0)

        return h_pred, expert_pred

    def predict_numpy(
        self,
        h: np.ndarray,
        z: np.ndarray,
        expert_routing: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Numpy wrapper."""
        h_t = torch.from_numpy(h).float().to(self._get_device())
        z_t = torch.from_numpy(z).float().to(self._get_device())
        exp_t = torch.from_numpy(expert_routing).float().to(self._get_device()) if expert_routing is not None else None

        with torch.no_grad():
            h_pred_t, exp_pred_t = self.forward(h_t, z_t, exp_t)

        return h_pred_t.cpu().numpy(), exp_pred_t.cpu().numpy() if exp_pred_t is not None else None

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"ActionPredictor saved to {path}")

    def load(self, path: Path):
        self.load_state_dict(torch.load(path, map_location=self._get_device()))
        self.eval()
        logger.info(f"ActionPredictor loaded from {path}")
