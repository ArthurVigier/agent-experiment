"""
integrator_moe.py — Geometric integrator avec analyse d'experts pour Qwen 3.5.

Étend l'intégrateur de base pour capturer et analyser les routages d'experts
dans l'architecture MoE.
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import json

from .scar_buffer import ScarBuffer
from .predictor_moe import MoEAwarePredictor

logger = logging.getLogger(__name__)


class MoEGeometricIntegrator:
    """
    Intégrateur géométrique avec conscience de l'architecture MoE.

    Capture en plus les routages d'experts et analyse leur relation
    avec les erreurs de prédiction.
    """

    def __init__(
        self,
        predictor: MoEAwarePredictor,
        scar_buffer: ScarBuffer,
        scar_threshold: float = 0.5,
        correction_strength: float = 1.0,
        num_experts: int = 8,
    ):
        """
        Args:
            predictor: MoEAwarePredictor entraîné
            scar_buffer: ScarBuffer instance
            scar_threshold: seuil de création de cicatrice
            correction_strength: force des corrections
            num_experts: nombre d'experts dans le modèle
        """
        self.predictor = predictor
        self.buffer = scar_buffer
        self.scar_threshold = scar_threshold
        self.correction_strength = correction_strength
        self.num_experts = num_experts

        self.stats = {
            "predictions": 0,
            "corrections": 0,
            "scars_created": 0,
            "total_error_before": 0.0,
            "total_error_after": 0.0,
            "expert_switches": 0,           # changements d'experts dominants
            "expert_error_correlation": {},  # corrélation erreur ↔ expert
        }

        # Matrice de suivi des experts
        self.expert_error_matrix = np.zeros((num_experts, num_experts))
        self.expert_counts = np.zeros(num_experts)

    def predict(
        self,
        h_before: np.ndarray,
        z_tool: np.ndarray,
        tool_name: str,
        expert_routing_before: Optional[np.ndarray] = None,
        return_correction: bool = True,
    ) -> Tuple[np.ndarray, Optional[list], Optional[np.ndarray]]:
        """
        Prédit le prochain état, avec correction et prédiction d'experts.

        Returns:
            h_pred: état latent prédit
            corrections: liste des scars utilisées
            expert_pred: distribution d'experts prédite
        """
        # Prédiction de base
        h_pred, expert_pred = self.predictor.predict_numpy(h_before, z_tool, expert_routing_before)
        self.stats["predictions"] += 1

        corrections = None
        if return_correction:
            # Requête au buffer
            h_corrected, corrections = self.buffer.correct_prediction(
                h_predicted=h_pred,
                h_context=h_before,
                z_tool=z_tool,
                tool_name=tool_name,
            )

            # Appliquer la correction
            if corrections:
                delta = h_corrected - h_pred
                h_pred = h_pred + delta * self.correction_strength
                self.stats["corrections"] += 1

        return h_pred, corrections, expert_pred

    def learn(
        self,
        h_before: np.ndarray,
        z_tool: np.ndarray,
        tool_name: str,
        h_predicted: np.ndarray,
        h_actual: np.ndarray,
        expert_routing_before: Optional[np.ndarray] = None,
        expert_routing_after: Optional[np.ndarray] = None,
        task_id: str = "",
    ):
        """
        Apprend de l'erreur, en analysant aussi les routages d'experts.
        """
        error_before = np.linalg.norm(h_actual - h_predicted)
        self.stats["total_error_before"] += error_before

        # Analyse des experts
        if expert_routing_before is not None and expert_routing_after is not None:
            # Expert dominant avant/après
            expert_before = np.argmax(expert_routing_before)
            expert_after = np.argmax(expert_routing_after)

            if expert_before != expert_after:
                self.stats["expert_switches"] += 1

            # Mise à jour de la matrice de corrélation erreur/experts
            self.expert_error_matrix[expert_before, expert_after] += error_before
            self.expert_counts[expert_before] += 1

        # Création de cicatrice si erreur significative
        if error_before >= self.scar_threshold:
            scar = self.buffer.record_error(
                h_context=h_before,
                z_tool=z_tool,
                tool_name=tool_name,
                h_predicted=h_predicted,
                h_actual=h_actual,
                task_id=task_id,
            )
            if scar:
                self.stats["scars_created"] += 1

        # Évaluation de la correction
        h_corrected, corrections = self.buffer.correct_prediction(
            h_predicted=h_predicted,
            h_context=h_before,
            z_tool=z_tool,
            tool_name=tool_name,
        )
        if corrections:
            error_after = np.linalg.norm(h_actual - h_corrected)
            self.stats["total_error_after"] += error_after
            self.buffer.report_correction_quality(h_predicted, h_corrected, h_actual)

    def get_expert_analysis(self) -> dict:
        """
        Analyse détaillée de la relation experts ↔ erreurs.
        """
        if self.expert_counts.sum() == 0:
            return {"error": "No expert data collected"}

        # Normaliser la matrice
        norm_matrix = np.zeros_like(self.expert_error_matrix)
        for i in range(self.num_experts):
            if self.expert_counts[i] > 0:
                norm_matrix[i, :] = self.expert_error_matrix[i, :] / self.expert_counts[i]

        # Identifier les paires expert→expert les plus erronées
        error_pairs = []
        for i in range(self.num_experts):
            for j in range(self.num_experts):
                if i != j and norm_matrix[i, j] > 0:
                    error_pairs.append({
                        "from_expert": int(i),
                        "to_expert": int(j),
                        "mean_error": float(norm_matrix[i, j]),
                        "count": int(self.expert_error_matrix[i, j] / (norm_matrix[i, j] + 1e-12)),
                    })

        error_pairs.sort(key=lambda x: -x["mean_error"])

        return {
            "expert_counts": self.expert_counts.tolist(),
            "error_matrix_norm": norm_matrix.tolist(),
            "top_error_transitions": error_pairs[:10],
            "total_expert_switches": self.stats["expert_switches"],
            "switch_rate": self.stats["expert_switches"] / max(1, self.stats["predictions"]),
        }

    def step(self):
        """Avance d'un pas."""
        self.buffer.step()

    def get_stats(self) -> dict:
        """Retourne les statistiques complètes."""
        stats = self.stats.copy()
        stats["buffer_stats"] = self.buffer.get_stats()
        stats["expert_analysis"] = self.get_expert_analysis()
        stats["mean_error_before"] = (
            self.stats["total_error_before"] / max(1, self.stats["predictions"])
        )
        stats["mean_error_after"] = (
            self.stats["total_error_after"] / max(1, self.stats["corrections"])
        )
        stats["error_reduction"] = (
            (stats["mean_error_before"] - stats["mean_error_after"])
            / max(1e-6, stats["mean_error_before"])
        )
        return stats

    def save(self, path: Path):
        """Sauvegarde l'état."""
        path.mkdir(parents=True, exist_ok=True)
        self.predictor.save(path / "predictor.pt")
        self.buffer.save(path / "buffer_stats.json")

        # Sauvegarder les stats experts
        with open(path / "expert_analysis.json", "w") as f:
            json.dump(self.get_expert_analysis(), f, indent=2)

        logger.info(f"MoEGeometricIntegrator saved to {path}")
