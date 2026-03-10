"""
integrator.py — Geometric integrator connecting predictor and scar buffer.
"""

import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path
import json

from .scar_buffer import ScarBuffer
from .predictor import ActionPredictor

logger = logging.getLogger(__name__)


class GeometricIntegrator:
    """
    Integrates action predictor and scar buffer into agent loop.
    """

    def __init__(
        self,
        predictor: ActionPredictor,
        scar_buffer: ScarBuffer,
        scar_threshold: float = 0.5,
        correction_strength: float = 1.0,
    ):
        self.predictor = predictor
        self.buffer = scar_buffer
        self.scar_threshold = scar_threshold
        self.correction_strength = correction_strength

        self.stats = {
            "predictions": 0,
            "corrections": 0,
            "scars_created": 0,
            "total_error_before": 0.0,
            "total_error_after": 0.0,
        }

    def predict(
        self,
        h_before: np.ndarray,
        z_tool: np.ndarray,
        tool_name: str,
        expert_routing: Optional[np.ndarray] = None,
        return_correction: bool = True,
    ) -> Tuple[np.ndarray, Optional[list]]:
        """Predict next state, optionally applying scar correction."""
        # Base prediction
        h_pred, _ = self.predictor.predict_numpy(h_before, z_tool, expert_routing)
        self.stats["predictions"] += 1

        corrections = None
        if return_correction:
            # Query buffer for corrections
            h_corrected, corrections = self.buffer.correct_prediction(
                h_predicted=h_pred,
                h_context=h_before,
                z_tool=z_tool,
                tool_name=tool_name,
            )

            # Apply correction strength
            if corrections:
                delta = h_corrected - h_pred
                h_pred = h_pred + delta * self.correction_strength
                self.stats["corrections"] += 1

        return h_pred, corrections

    def learn(
        self,
        h_before: np.ndarray,
        z_tool: np.ndarray,
        tool_name: str,
        h_predicted: np.ndarray,
        h_actual: np.ndarray,
        task_id: str = "",
    ):
        """Learn from actual outcome after tool execution."""
        error_before = np.linalg.norm(h_actual - h_predicted)
        self.stats["total_error_before"] += error_before

        # Create scar if error significant
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

        # Check if correction would have helped
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

    def step(self):
        """Advance one time step (decay scars)."""
        self.buffer.step()

    def get_stats(self) -> dict:
        """Return integrator statistics."""
        stats = self.stats.copy()
        stats["buffer_stats"] = self.buffer.get_stats()
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
        """Save integrator state."""
        path.mkdir(parents=True, exist_ok=True)
        self.predictor.save(path / "predictor.pt")
        self.buffer.save(path / "buffer_stats.json")
        logger.info(f"GeometricIntegrator saved to {path}")

    def reset_task(self):
        """Reset between tasks (optional)."""
        self.buffer.reset_cross_task(decay_factor=0.5)
