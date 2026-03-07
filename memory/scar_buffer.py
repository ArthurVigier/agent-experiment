"""
scar_buffer.py
Mémoire géométrique par cicatrices (scars) dans l'espace latent.

Quand un tool call produit un résultat inattendu, le vecteur d'erreur
δ = h_réel - ĥ_prédit est stocké comme une "cicatrice" dans l'espace latent.
Les cicatrices déforment les prédictions futures pour les états voisins,
permettant à l'agent d'apprendre de ses erreurs pendant le run.

Trois niveaux de généralisation :
  1. Répétition directe (cos > 0.95) : même situation → même correction
  2. Généralisation locale (cos > 0.80) : situation analogue → correction pondérée
  3. Abstraction (direction commune des δ) : pattern d'erreur → règle implicite

Convention : docstrings français, code anglais.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Scar:
    """Une cicatrice dans l'espace latent."""
    h_context: np.ndarray       # état latent au moment de l'erreur
    z_tool: np.ndarray          # embedding du tool (ou nom si pas d'embedding)
    tool_name: str              # nom du tool pour le debug
    delta: np.ndarray           # h_réel - ĥ_prédit (vecteur d'erreur)
    magnitude: float            # ||delta|| — gravité de l'erreur
    step_created: int           # step où la cicatrice a été créée
    task_id: str                # tâche d'origine
    decay: float = 1.0          # facteur de décroissance (diminue avec le temps)
    hit_count: int = 0          # nombre de fois que cette scar a été activée


@dataclass
class ScarMatch:
    """Résultat d'une requête au scar buffer."""
    scar: Scar
    similarity: float           # cosinus entre l'état courant et la scar
    weighted_delta: np.ndarray  # correction pondérée par la similarité et le decay
    influence: float            # poids total de cette scar sur la correction


class ScarBuffer:
    """
    Buffer de cicatrices pour la mémoire géométrique.
    
    Le buffer stocke les erreurs de prédiction et les utilise pour
    corriger les prédictions futures dans le voisinage géométrique.
    """

    def __init__(
        self,
        max_size: int = 64,
        similarity_threshold: float = 0.80,
        magnitude_threshold: float = 0.1,
        decay_rate: float = 0.95,
        kernel_bandwidth: float = 0.1,
        tool_match_required: bool = False,
    ):
        """
        Args:
            max_size: nombre max de cicatrices stockées
            similarity_threshold: cosinus minimum pour qu'une scar influence
            magnitude_threshold: ||δ|| minimum pour créer une scar
            decay_rate: facteur de décroissance par step (decay *= decay_rate chaque step)
            kernel_bandwidth: paramètre du kernel gaussien (σ)
            tool_match_required: si True, la scar n'influence que le même tool
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.magnitude_threshold = magnitude_threshold
        self.decay_rate = decay_rate
        self.kernel_bandwidth = kernel_bandwidth
        self.tool_match_required = tool_match_required

        self.scars: list[Scar] = []
        self.current_step: int = 0
        self.stats = {
            "scars_created": 0,
            "scars_evicted": 0,
            "queries": 0,
            "hits": 0,
            "corrections_applied": 0,
            "false_positives": 0,  # corrections qui empirent le résultat
        }

    def record_error(
        self,
        h_context: np.ndarray,
        z_tool: np.ndarray,
        tool_name: str,
        h_predicted: np.ndarray,
        h_actual: np.ndarray,
        task_id: str = "",
    ) -> Optional[Scar]:
        """
        Enregistre une erreur de prédiction si elle est significative.
        
        Args:
            h_context: état latent au moment du tool call
            z_tool: embedding du tool (ou vecteur nul si pas d'embedding)
            tool_name: nom du tool
            h_predicted: ĥ prédit par le prédicteur
            h_actual: h réel après exécution
            task_id: identifiant de la tâche
            
        Returns:
            La Scar créée, ou None si l'erreur est trop petite
        """
        delta = h_actual - h_predicted
        magnitude = float(np.linalg.norm(delta))

        if magnitude < self.magnitude_threshold:
            return None  # Erreur trop petite, pas de scar

        scar = Scar(
            h_context=h_context.copy(),
            z_tool=z_tool.copy() if z_tool is not None else np.zeros(1),
            tool_name=tool_name,
            delta=delta.copy(),
            magnitude=magnitude,
            step_created=self.current_step,
            task_id=task_id,
        )

        self._insert(scar)
        self.stats["scars_created"] += 1

        logger.debug(f"Scar created: tool={tool_name}, magnitude={magnitude:.4f}, "
                     f"buffer_size={len(self.scars)}")

        return scar

    def query(
        self,
        h_context: np.ndarray,
        z_tool: Optional[np.ndarray] = None,
        tool_name: Optional[str] = None,
    ) -> list[ScarMatch]:
        """
        Cherche les scars pertinentes pour l'état courant.
        
        Returns:
            Liste de ScarMatch ordonnée par influence décroissante
        """
        self.stats["queries"] += 1

        if not self.scars:
            return []

        h_norm = h_context / (np.linalg.norm(h_context) + 1e-12)
        matches = []

        for scar in self.scars:
            # Match de tool si requis
            if self.tool_match_required and tool_name and scar.tool_name != tool_name:
                continue

            # Similarité cosinus
            scar_h_norm = scar.h_context / (np.linalg.norm(scar.h_context) + 1e-12)
            cos_sim = float(np.dot(h_norm, scar_h_norm))

            if cos_sim < self.similarity_threshold:
                continue

            # Kernel gaussien : influence décroît avec la distance
            distance = 1.0 - cos_sim
            kernel_weight = np.exp(-0.5 * (distance / self.kernel_bandwidth) ** 2)

            # Influence totale : kernel × decay × magnitude
            influence = kernel_weight * scar.decay * scar.magnitude

            # Correction pondérée
            weighted_delta = scar.delta * kernel_weight * scar.decay

            matches.append(ScarMatch(
                scar=scar,
                similarity=cos_sim,
                weighted_delta=weighted_delta,
                influence=influence,
            ))

        # Trier par influence
        matches.sort(key=lambda m: m.influence, reverse=True)

        if matches:
            self.stats["hits"] += 1

        return matches

    def correct_prediction(
        self,
        h_predicted: np.ndarray,
        h_context: np.ndarray,
        z_tool: Optional[np.ndarray] = None,
        tool_name: Optional[str] = None,
        max_corrections: int = 5,
    ) -> tuple[np.ndarray, list[ScarMatch]]:
        """
        Corrige une prédiction en utilisant les scars pertinentes.
        
        Args:
            h_predicted: prédiction originale du prédicteur
            h_context: état latent courant
            z_tool: embedding du tool candidat
            tool_name: nom du tool candidat
            max_corrections: nombre max de scars à utiliser
            
        Returns:
            (h_corrigé, liste des scars utilisées)
        """
        matches = self.query(h_context, z_tool, tool_name)

        if not matches:
            return h_predicted, []

        # Appliquer les corrections (somme pondérée des deltas)
        corrections = matches[:max_corrections]
        total_delta = np.zeros_like(h_predicted)
        for match in corrections:
            total_delta += match.weighted_delta
            match.scar.hit_count += 1

        h_corrected = h_predicted + total_delta
        self.stats["corrections_applied"] += 1

        logger.debug(f"Prediction corrected: {len(corrections)} scars, "
                     f"||correction||={np.linalg.norm(total_delta):.4f}")

        return h_corrected, corrections

    def step(self):
        """Avance d'un step : décroissance de toutes les scars."""
        self.current_step += 1
        for scar in self.scars:
            scar.decay *= self.decay_rate

    def report_correction_quality(
        self,
        h_predicted_original: np.ndarray,
        h_predicted_corrected: np.ndarray,
        h_actual: np.ndarray,
    ):
        """
        Enregistre si une correction a amélioré ou empiré la prédiction.
        Utile pour calibrer le rayon et le decay.
        """
        error_before = float(np.linalg.norm(h_actual - h_predicted_original))
        error_after = float(np.linalg.norm(h_actual - h_predicted_corrected))

        if error_after > error_before:
            self.stats["false_positives"] += 1
            logger.debug(f"False positive: correction worsened prediction "
                         f"(error {error_before:.4f} → {error_after:.4f})")

    def _insert(self, scar: Scar):
        """Insère une scar, éjecte la plus faible si le buffer est plein."""
        if len(self.scars) >= self.max_size:
            # Éjecter la scar avec le plus faible decay × magnitude
            weakest_idx = min(
                range(len(self.scars)),
                key=lambda i: self.scars[i].decay * self.scars[i].magnitude,
            )
            self.scars.pop(weakest_idx)
            self.stats["scars_evicted"] += 1

        self.scars.append(scar)

    def clear(self):
        """Vide le buffer (entre tâches si on ne veut pas de transfert)."""
        self.scars.clear()

    def get_stats(self) -> dict:
        """Retourne les statistiques du buffer."""
        active_scars = [s for s in self.scars if s.decay > 0.01]
        return {
            **self.stats,
            "buffer_size": len(self.scars),
            "active_scars": len(active_scars),
            "mean_magnitude": float(np.mean([s.magnitude for s in self.scars])) if self.scars else 0,
            "mean_decay": float(np.mean([s.decay for s in self.scars])) if self.scars else 0,
            "mean_hit_count": float(np.mean([s.hit_count for s in self.scars])) if self.scars else 0,
            "false_positive_rate": (
                self.stats["false_positives"] / max(1, self.stats["corrections_applied"])
            ),
            "hit_rate": self.stats["hits"] / max(1, self.stats["queries"]),
        }

    def analyze_scars(self) -> dict:
        """
        Analyse la topologie des cicatrices.
        
        Retourne des métriques utiles pour le diagnostic :
        - PCA sur les deltas → directions principales d'erreur
        - Clustering des scars → catégories d'erreurs
        - Statistiques par tool
        """
        if len(self.scars) < 3:
            return {"n_scars": len(self.scars), "insufficient_data": True}

        deltas = np.stack([s.delta for s in self.scars])
        contexts = np.stack([s.h_context for s in self.scars])

        # PCA sur les deltas
        from sklearn.decomposition import PCA
        n_components = min(5, len(deltas) - 1, deltas.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(deltas)

        # Stats par tool
        tool_stats = {}
        for scar in self.scars:
            if scar.tool_name not in tool_stats:
                tool_stats[scar.tool_name] = {"count": 0, "mean_magnitude": 0, "hit_count": 0}
            tool_stats[scar.tool_name]["count"] += 1
            tool_stats[scar.tool_name]["mean_magnitude"] += scar.magnitude
            tool_stats[scar.tool_name]["hit_count"] += scar.hit_count
        for tool in tool_stats:
            n = tool_stats[tool]["count"]
            tool_stats[tool]["mean_magnitude"] /= n
            tool_stats[tool]["hit_count"] /= n

        return {
            "n_scars": len(self.scars),
            "delta_pca_variance": pca.explained_variance_ratio_.tolist(),
            "dominant_error_direction_variance": float(pca.explained_variance_ratio_[0]),
            "tool_stats": tool_stats,
        }

    def save(self, path: Path):
        """Sauvegarde le buffer pour analyse offline."""
        data = {
            "stats": self.get_stats(),
            "analysis": self.analyze_scars(),
            "scars": [
                {
                    "tool_name": s.tool_name,
                    "magnitude": s.magnitude,
                    "step_created": s.step_created,
                    "task_id": s.task_id,
                    "decay": s.decay,
                    "hit_count": s.hit_count,
                }
                for s in self.scars
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
