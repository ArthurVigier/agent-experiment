"""
predictor_moe.py — Action predictor optimisé pour l'architecture MoE de Qwen 3.5.

Tient compte de la structure Mixture of Experts pour mieux prédire
les transitions d'état dans l'espace latent.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class MoEAwarePredictor(nn.Module):
    """
    ActionPredictor qui prend en compte la structure MoE de Qwen 3.5.

    Caractéristiques:
      - Prédit non seulement le prochain état, mais aussi la distribution d'experts
      - Utilise l'information des experts pour améliorer la prédiction
      - Peut détecter quand un outil nécessite un expert différent
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int = 128,
        num_experts: int = 8,  # Par couche MoE typique
        hidden_sizes: list[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: dimension des états latents (généralement 4096 pour 8B, 5120 pour 35B)
            embed_dim: dimension des embeddings d'outils
            num_experts: nombre d'experts dans les couches MoE
            hidden_sizes: tailles des couches cachées du MLP
        """
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [1024, 1024]  # Plus grand pour capturer la complexité MoE

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.input_dim = hidden_dim + embed_dim + num_experts  # On ajoute l'info des experts

        # Encodeur d'experts (optionnel, si on veut apprendre une représentation)
        self.expert_encoder = nn.Sequential(
            nn.Linear(num_experts, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        # MLP principal avec attention sur les experts
        self.expert_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Build MLP
        layers = []
        prev_dim = hidden_dim + embed_dim + 32  # hidden + z + expert_encoding
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, size),
                nn.LayerNorm(size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = size
        layers.append(nn.Linear(prev_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

        # Prédicteur de distribution d'experts pour l'état suivant
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
            h: (batch, hidden_dim) état latent courant
            z: (batch, embed_dim) embedding de l'outil
            expert_routing: (batch, num_experts) distribution des experts activés
                            (optionnel, None si non disponible)

        Returns:
            h_pred: (batch, hidden_dim) état latent prédit
            expert_pred: (batch, num_experts) distribution d'experts prédite
        """
        # Handle single samples
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Ensure same batch size
        if h.size(0) != z.size(0):
            if h.size(0) == 1:
                h = h.expand(z.size(0), -1)
            elif z.size(0) == 1:
                z = z.expand(h.size(0), -1)
            else:
                raise ValueError(f"Batch size mismatch: h={h.size(0)}, z={z.size(0)}")

        # Encode expert routing if available
        expert_features = None
        if expert_routing is not None:
            if expert_routing.dim() == 1:
                expert_routing = expert_routing.unsqueeze(0)
            expert_features = self.expert_encoder(expert_routing.float())

        # Apply expert attention (self-attention on h)
        h_attended, _ = self.expert_attention(h, h, h)

        # Concatenate features
        concat_inputs = [h_attended, z]
        if expert_features is not None:
            concat_inputs.append(expert_features)

        x = torch.cat(concat_inputs, dim=-1)
        h_pred = self.mlp(x)

        # Predict next expert distribution
        expert_pred = self.expert_predictor(h_pred)

        if h_pred.size(0) == 1:
            h_pred = h_pred.squeeze(0)
            if expert_pred.size(0) == 1:
                expert_pred = expert_pred.squeeze(0)

        return h_pred, expert_pred

    def predict_numpy(
        self,
        h: np.ndarray,
        z: np.ndarray,
        expert_routing: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Numpy wrapper for forward pass."""
        h_t = torch.from_numpy(h).float().to(self._get_device())
        z_t = torch.from_numpy(z).float().to(self._get_device())
        exp_t = torch.from_numpy(expert_routing).float().to(self._get_device()) if expert_routing is not None else None

        with torch.no_grad():
            h_pred_t, exp_pred_t = self.forward(h_t, z_t, exp_t)

        return h_pred_t.cpu().numpy(), exp_pred_t.cpu().numpy() if exp_pred_t is not None else None

    def save(self, path: Path):
        """Save model weights."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"MoEAwarePredictor saved to {path}")

    def load(self, path: Path):
        """Load model weights."""
        self.load_state_dict(torch.load(path, map_location=self._get_device()))
        self.eval()
        logger.info(f"MoEAwarePredictor loaded from {path}")
