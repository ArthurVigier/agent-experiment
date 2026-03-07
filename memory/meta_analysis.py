"""
meta_analysis.py
Instrumentation pour l'analyse récursive (poupée russe).

PAS un composant actif de l'agent. Un outil de collecte de métriques
qui se branche sur les modules existants et enregistre les signaux
nécessaires pour une future analyse méta-récursive.

Coût pendant le run : quasi-nul (quelques np.save et des moyennes).
Coût de l'analyse post-hoc : 10-30 min de compute sur les données sauvegardées.
Coût du stress-test : 1-2h de compute (optionnel, séparé du run principal).

Hiérarchie des niveaux :
  Niveau 0 : LLM → hidden states → Â, routing, simulation
  Niveau 1 : Prédicteur JEPA → activations internes → erreurs structurées
  Niveau 2 : Scar buffer → topologie des cicatrices → patterns d'erreur
  Niveau 3 : Méta-Â → le prédicteur sait-il quand il va se tromper ?
  Niveau N : récursion → quand le signal devient du bruit

Convention : docstrings français, code anglais.
"""

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Collecteur de métriques méta (coût : ~0 pendant le run)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PredictorErrorRecord:
    """Enregistrement d'une erreur de prédiction pour analyse méta."""
    step: int
    task_id: str
    tool_name: str
    # Vecteurs d'erreur
    delta: Optional[np.ndarray] = None         # h_réel - ĥ_prédit
    delta_norm: float = 0.0                     # ||delta||
    cosine_pred_vs_real: float = 0.0            # cos(ĥ, h_réel)
    # Activations du prédicteur (si disponibles)
    predictor_hidden: Optional[np.ndarray] = None  # activations internes du prédicteur
    # Confiance pré-exécution
    predictor_confidence: float = 0.0           # métrique de confiance avant exécution
    # Résultat
    was_useful: bool = True                     # le tool call a-t-il contribué au succès


class MetaCollector:
    """
    Collecteur passif de métriques pour l'analyse récursive.
    
    S'accroche aux modules existants via des callbacks.
    N'ajoute aucun compute au chemin critique.
    Sauvegarde périodiquement pour analyse post-hoc.
    """

    def __init__(self, output_dir: Path, save_vectors: bool = True):
        """
        Args:
            output_dir: répertoire de sortie pour les métriques
            save_vectors: si True, sauvegarde les vecteurs d'erreur (plus gros mais
                         nécessaire pour PCA/analyse géométrique). Si False, ne
                         sauvegarde que les scalaires (léger).
        """
        self.output_dir = output_dir / "meta"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_vectors = save_vectors

        self.error_records: list[PredictorErrorRecord] = []
        self.scar_snapshots: list[dict] = []  # snapshots périodiques du scar buffer
        self.confidence_calibration: list[tuple[float, float]] = []  # (confiance, erreur_réelle)

    # ── Callbacks à brancher sur les modules existants ──

    def on_prediction(
        self,
        step: int,
        task_id: str,
        tool_name: str,
        h_predicted: np.ndarray,
        h_actual: np.ndarray,
        predictor_hidden: Optional[np.ndarray] = None,
        predictor_confidence: float = 0.0,
        was_useful: bool = True,
    ):
        """
        Appelé après chaque tool call, quand on a la prédiction ET le résultat.
        Coût : une soustraction + une norme + un cosinus = ~0.
        """
        delta = h_actual - h_predicted
        delta_norm = float(np.linalg.norm(delta))
        cos_sim = float(
            np.dot(h_predicted, h_actual)
            / (np.linalg.norm(h_predicted) * np.linalg.norm(h_actual) + 1e-12)
        )

        record = PredictorErrorRecord(
            step=step,
            task_id=task_id,
            tool_name=tool_name,
            delta=delta if self.save_vectors else None,
            delta_norm=delta_norm,
            cosine_pred_vs_real=cos_sim,
            predictor_hidden=predictor_hidden if self.save_vectors else None,
            predictor_confidence=predictor_confidence,
            was_useful=was_useful,
        )
        self.error_records.append(record)
        self.confidence_calibration.append((predictor_confidence, delta_norm))

    def on_scar_buffer_snapshot(self, scar_stats: dict, step: int):
        """Appelé périodiquement pour logger l'état du scar buffer."""
        self.scar_snapshots.append({"step": step, **scar_stats})

    # ── Sauvegarde ──

    def save(self):
        """Sauvegarde toutes les métriques collectées."""
        # Scalaires
        scalars = [{
            "step": r.step,
            "task_id": r.task_id,
            "tool_name": r.tool_name,
            "delta_norm": r.delta_norm,
            "cosine_pred_vs_real": r.cosine_pred_vs_real,
            "predictor_confidence": r.predictor_confidence,
            "was_useful": r.was_useful,
        } for r in self.error_records]

        with open(self.output_dir / "prediction_errors.json", "w") as f:
            json.dump(scalars, f, indent=2)

        # Vecteurs (si activés)
        if self.save_vectors:
            deltas = [r.delta for r in self.error_records if r.delta is not None]
            if deltas:
                np.save(self.output_dir / "error_deltas.npy", np.stack(deltas))
            
            predictor_hiddens = [r.predictor_hidden for r in self.error_records
                                 if r.predictor_hidden is not None]
            if predictor_hiddens:
                np.save(self.output_dir / "predictor_hiddens.npy", np.stack(predictor_hiddens))

        # Scar buffer evolution
        if self.scar_snapshots:
            with open(self.output_dir / "scar_evolution.json", "w") as f:
                json.dump(self.scar_snapshots, f, indent=2)

        # Calibration
        if self.confidence_calibration:
            cal = np.array(self.confidence_calibration)
            np.save(self.output_dir / "calibration.npy", cal)

        logger.info(f"Meta metrics saved: {len(self.error_records)} predictions, "
                     f"{len(self.scar_snapshots)} scar snapshots")


# ═══════════════════════════════════════════════════════════════════════════
# Analyse post-hoc (se lance APRÈS le run, sur les données sauvegardées)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_level1_predictor_errors(meta_dir: Path) -> dict:
    """
    Niveau 1 : structure des erreurs du prédicteur.
    
    Questions :
    - Les erreurs ont-elles une direction principale ? (PCA sur les deltas)
    - Les erreurs clustent-elles par tool ? (silhouette)
    - Les erreurs diminuent-elles au cours du run ? (learning curve)
    """
    results = {}

    # Charger les données
    errors_path = meta_dir / "prediction_errors.json"
    deltas_path = meta_dir / "error_deltas.npy"

    if not errors_path.exists():
        return {"error": "no prediction errors found"}

    with open(errors_path) as f:
        errors = json.load(f)

    results["n_predictions"] = len(errors)
    results["mean_error"] = float(np.mean([e["delta_norm"] for e in errors]))
    results["mean_cosine"] = float(np.mean([e["cosine_pred_vs_real"] for e in errors]))

    # Learning curve : erreur moyenne par fenêtre glissante
    norms = [e["delta_norm"] for e in errors]
    window = max(5, len(norms) // 10)
    if len(norms) >= window * 2:
        learning_curve = [
            float(np.mean(norms[i:i + window]))
            for i in range(0, len(norms) - window + 1, window)
        ]
        results["learning_curve"] = learning_curve
        results["error_trend"] = "improving" if learning_curve[-1] < learning_curve[0] * 0.8 else "stable"

    # Erreur par tool
    from collections import defaultdict
    tool_errors = defaultdict(list)
    for e in errors:
        tool_errors[e["tool_name"]].append(e["delta_norm"])
    results["error_by_tool"] = {
        tool: {"mean": float(np.mean(errs)), "std": float(np.std(errs)), "n": len(errs)}
        for tool, errs in tool_errors.items()
    }

    # PCA sur les deltas (si disponibles)
    if deltas_path.exists():
        deltas = np.load(deltas_path)
        if len(deltas) >= 5:
            from sklearn.decomposition import PCA
            n_comp = min(10, len(deltas) - 1, deltas.shape[1])
            pca = PCA(n_components=n_comp)
            pca.fit(deltas)
            results["error_pca_variance"] = pca.explained_variance_ratio_.tolist()
            results["dominant_error_direction_variance"] = float(pca.explained_variance_ratio_[0])

            # Si PC1 explique > 40% → il y a une direction d'erreur dominante
            if pca.explained_variance_ratio_[0] > 0.4:
                results["dominant_error_interpretation"] = (
                    "Le prédicteur fait systématiquement le même type d'erreur. "
                    "Cette direction pourrait être intégrée comme biais correctif."
                )

    return results


def analyze_level2_scar_topology(meta_dir: Path) -> dict:
    """
    Niveau 2 : évolution et topologie du scar buffer.
    
    Questions :
    - Les scars convergent-elles vers un état stable ? (entropie au cours du temps)
    - Les scars les plus anciennes sont-elles les plus utilisées ? (corrélation âge/hits)
    - Le buffer est-il sous-utilisé (trop restrictif) ou sur-utilisé (trop permissif) ?
    """
    scar_path = meta_dir / "scar_evolution.json"
    if not scar_path.exists():
        return {"error": "no scar evolution data"}

    with open(scar_path) as f:
        snapshots = json.load(f)

    if not snapshots:
        return {"error": "empty scar evolution"}

    results = {
        "n_snapshots": len(snapshots),
        "buffer_size_evolution": [s.get("buffer_size", 0) for s in snapshots],
        "hit_rate_evolution": [s.get("hit_rate", 0) for s in snapshots],
        "false_positive_rate_evolution": [s.get("false_positive_rate", 0) for s in snapshots],
    }

    # Convergence : est-ce que le hit rate se stabilise ?
    hit_rates = results["hit_rate_evolution"]
    if len(hit_rates) >= 4:
        first_half = np.mean(hit_rates[:len(hit_rates) // 2])
        second_half = np.mean(hit_rates[len(hit_rates) // 2:])
        results["hit_rate_trend"] = "stabilizing" if abs(second_half - first_half) < 0.05 else "evolving"

    # Diagnostic d'utilisation
    final = snapshots[-1]
    hit_rate = final.get("hit_rate", 0)
    fp_rate = final.get("false_positive_rate", 0)

    if hit_rate < 0.05:
        results["buffer_diagnosis"] = "under_utilized — rayon trop restrictif ou peu d'erreurs récurrentes"
    elif hit_rate > 0.40 and fp_rate > 0.20:
        results["buffer_diagnosis"] = "over_utilized — rayon trop large, beaucoup de faux positifs"
    elif fp_rate < 0.10 and hit_rate > 0.10:
        results["buffer_diagnosis"] = "well_calibrated — bon ratio signal/bruit"
    else:
        results["buffer_diagnosis"] = "marginal — impact limité"

    return results


def analyze_level3_meta_confidence(meta_dir: Path) -> dict:
    """
    Niveau 3 : le prédicteur sait-il quand il va se tromper ?
    
    Corrélation entre la confiance pré-exécution et l'erreur réelle.
    Si la corrélation est forte → le prédicteur a un signal de méta-Â intrinsèque.
    Si faible → le prédicteur est aveugle à sa propre fiabilité.
    """
    cal_path = meta_dir / "calibration.npy"
    if not cal_path.exists():
        return {"error": "no calibration data"}

    cal = np.load(cal_path)  # (N, 2) : confidence, error
    if len(cal) < 10:
        return {"error": "insufficient data", "n": len(cal)}

    confidence = cal[:, 0]
    errors = cal[:, 1]

    # Spearman correlation (robuste aux non-linéarités)
    from scipy.stats import spearmanr
    rho, p_value = spearmanr(confidence, errors)

    # Binned calibration : confiance haute → erreur basse ?
    n_bins = min(5, len(cal) // 5)
    if n_bins >= 2:
        sorted_idx = np.argsort(confidence)
        bin_size = len(cal) // n_bins
        bins = []
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(cal)
            bin_conf = float(np.mean(confidence[sorted_idx[start:end]]))
            bin_err = float(np.mean(errors[sorted_idx[start:end]]))
            bins.append({"mean_confidence": bin_conf, "mean_error": bin_err})
    else:
        bins = []

    results = {
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_value),
        "calibration_bins": bins,
        "n_predictions": len(cal),
    }

    # Verdict
    if abs(rho) > 0.5 and p_value < 0.01:
        results["verdict"] = (
            "meta_calibrated — le prédicteur SAIT quand il va se tromper. "
            "Signal de méta-Â exploitable pour le gating de la simulation."
        )
    elif abs(rho) > 0.3:
        results["verdict"] = "partially_calibrated — signal partiel de méta-confiance"
    else:
        results["verdict"] = "uncalibrated — le prédicteur est aveugle à sa fiabilité"

    return results


def run_full_meta_analysis(meta_dir: Path) -> dict:
    """Lance les trois niveaux d'analyse méta et produit un rapport."""
    results = {
        "level1_predictor_errors": analyze_level1_predictor_errors(meta_dir),
        "level2_scar_topology": analyze_level2_scar_topology(meta_dir),
        "level3_meta_confidence": analyze_level3_meta_confidence(meta_dir),
    }

    # Sauvegarder
    with open(meta_dir / "meta_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Rapport
    print("\n" + "=" * 70)
    print("ANALYSE MÉTA-RÉCURSIVE (Poupée Russe)")
    print("=" * 70)

    l1 = results["level1_predictor_errors"]
    if "error" not in l1:
        print(f"\n─── Niveau 1 : Erreurs du prédicteur ───")
        print(f"  Erreur moyenne : {l1['mean_error']:.4f}")
        print(f"  Cosinus moyen  : {l1['mean_cosine']:.4f}")
        if "error_trend" in l1:
            print(f"  Tendance       : {l1['error_trend']}")
        if "dominant_error_direction_variance" in l1:
            var1 = l1["dominant_error_direction_variance"]
            print(f"  PC1 variance   : {var1:.3f}")
            if var1 > 0.4:
                print(f"  ⚠ Direction d'erreur dominante détectée — candidate pour biais correctif")

    l2 = results["level2_scar_topology"]
    if "error" not in l2:
        print(f"\n─── Niveau 2 : Topologie du scar buffer ───")
        print(f"  Diagnostic : {l2.get('buffer_diagnosis', 'n/a')}")
        if "hit_rate_trend" in l2:
            print(f"  Tendance hit rate : {l2['hit_rate_trend']}")

    l3 = results["level3_meta_confidence"]
    if "error" not in l3:
        print(f"\n─── Niveau 3 : Méta-confiance ───")
        print(f"  Spearman ρ : {l3['spearman_rho']:.3f} (p={l3['spearman_p_value']:.4f})")
        print(f"  Verdict    : {l3['verdict']}")

    print("=" * 70)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Stress-test de profondeur récursive (OPTIONNEL, séparé du run principal)
# ═══════════════════════════════════════════════════════════════════════════

def stress_test_recursion_depth(
    error_deltas_path: Path,
    max_depth: int = 6,
    min_samples: int = 20,
) -> dict:
    """
    Stress-test : empile les niveaux d'analyse et mesure quand le signal
    devient du bruit.
    
    Protocole :
      Niveau 0 : PCA sur les hidden states → directions principales
      Niveau 1 : PCA sur les erreurs de prédiction → directions d'erreur
      Niveau 2 : PCA sur les erreurs des erreurs (résidus de la reconstruction PCA L1)
      Niveau N : PCA sur les résidus du niveau N-1
    
    À chaque niveau, on mesure :
      - Variance expliquée par PC1 (si > 0.3 → il y a du signal structuré)
      - Silhouette (si > 0.1 → il y a des clusters)
      - AUC de discrimination (si > 0.6 → le signal est exploitable)
    
    La profondeur utile est le dernier niveau où PC1 variance > 0.3.
    
    Usage :
      python -c "from memory.meta_analysis import stress_test_recursion_depth; \\
                  stress_test_recursion_depth(Path('results/sprint2/meta/error_deltas.npy'))"
    
    Coût : ~5 min sur les données déjà collectées (aucun forward pass).
    """
    if not error_deltas_path.exists():
        return {"error": f"File not found: {error_deltas_path}"}

    data = np.load(error_deltas_path)
    if len(data) < min_samples:
        return {"error": f"Insufficient data: {len(data)} < {min_samples}"}

    from sklearn.decomposition import PCA

    results = {"levels": [], "useful_depth": 0}
    current_data = data

    for depth in range(max_depth):
        n_samples, n_dims = current_data.shape

        if n_samples < min_samples or n_dims < 3:
            results["levels"].append({
                "depth": depth,
                "status": "terminated",
                "reason": f"insufficient data: {n_samples} samples, {n_dims} dims",
            })
            break

        # PCA
        n_comp = min(10, n_samples - 1, n_dims)
        pca = PCA(n_components=n_comp)
        projected = pca.fit_transform(current_data)
        pc1_var = float(pca.explained_variance_ratio_[0])
        total_var_top3 = float(sum(pca.explained_variance_ratio_[:3]))

        # Reconstruction + résidus
        reconstructed = pca.inverse_transform(projected)
        residuals = current_data - reconstructed

        # Norme des résidus vs norme des données
        data_norm = float(np.mean(np.linalg.norm(current_data, axis=1)))
        residual_norm = float(np.mean(np.linalg.norm(residuals, axis=1)))
        signal_to_noise = data_norm / (residual_norm + 1e-12)

        level_result = {
            "depth": depth,
            "n_samples": n_samples,
            "n_dims": n_dims,
            "pc1_variance": pc1_var,
            "top3_variance": total_var_top3,
            "data_norm": data_norm,
            "residual_norm": residual_norm,
            "signal_to_noise": signal_to_noise,
            "has_structure": pc1_var > 0.3,
        }
        results["levels"].append(level_result)

        if pc1_var > 0.3:
            results["useful_depth"] = depth + 1

        # Passer au niveau suivant : les résidus deviennent les données
        current_data = residuals

        # Si le SNR est < 1.1 → plus de signal, que du bruit
        if signal_to_noise < 1.1:
            results["levels"][-1]["status"] = "noise_floor"
            break

    # Rapport
    print("\n" + "=" * 70)
    print("STRESS-TEST PROFONDEUR RÉCURSIVE")
    print("=" * 70)

    for level in results["levels"]:
        d = level["depth"]
        pc1 = level.get("pc1_variance", 0)
        snr = level.get("signal_to_noise", 0)
        has = "✓" if level.get("has_structure") else "✗"
        status = level.get("status", "")
        print(f"  Niveau {d} : PC1={pc1:.3f}  SNR={snr:.2f}  structure={has}  {status}")

    print(f"\n  PROFONDEUR UTILE : {results['useful_depth']} niveaux")
    if results["useful_depth"] >= 3:
        print(f"  → La récursion méta est productive sur ≥3 niveaux")
        print(f"  → L'analyse de la poupée russe est JUSTIFIÉE")
    elif results["useful_depth"] >= 2:
        print(f"  → Signal méta au niveau 1, bruit au-delà")
        print(f"  → Analyse des erreurs du prédicteur utile, pas de récursion profonde")
    else:
        print(f"  → Pas de signal méta exploitable")
        print(f"  → Les erreurs du prédicteur sont du bruit non structuré")

    print("=" * 70)

    return results
