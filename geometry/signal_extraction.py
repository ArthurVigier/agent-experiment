"""
signal_extraction.py
Extraction de signaux multiples pour la cascade Sprint 0C-bis.

Capture en un seul forward pass :
  - Hidden states à plusieurs couches (sweep)
  - Mean pooling ET last-token
  - Logits top-k au dernier token
  - Entropie des logits
  - Décomposition attention/FFN (optionnel, coûteux)

Conçu pour s'intégrer dans le HiddenStateLogger existant
sans modifier la boucle de l'agent.

Convention : docstrings français, code anglais.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class StepSignals:
    """Tous les signaux extraits d'un step."""
    # Hidden states à différentes profondeurs
    hidden_states: dict[str, np.ndarray] = field(default_factory=dict)
    # Format des clés : "layer{L}_{pooling}" e.g. "layer16_mean", "layer16_last"

    # Logits au dernier token
    logits_topk_tokens: list[str] = field(default_factory=list)    # noms des top-k tokens
    logits_topk_probs: list[float] = field(default_factory=list)   # probabilités
    logits_entropy: float = 0.0                                     # entropie de la distribution
    logits_top1_prob: float = 0.0                                   # prob du token le plus probable

    # Metadata
    step_idx: int = 0
    has_tool_call: bool = False
    tool_name: Optional[str] = None


class MultiSignalLogger:
    """
    Logger enrichi qui capture plusieurs signaux en un seul forward pass.
    
    Remplace le HiddenStateLogger simple pour Sprint 0C-bis.
    Compatible avec le même interface (get_last_hidden_state, save).
    """

    def __init__(
        self,
        model,
        tokenizer,
        num_layers: int,
        output_dir: Path,
        layer_fractions: list[float] = None,
        topk: int = 50,
        capture_logits: bool = True,
        capture_decomposition: bool = False,
    ):
        """
        Args:
            model: le modèle HuggingFace
            tokenizer: le tokenizer
            num_layers: nombre total de couches
            output_dir: répertoire de sortie
            layer_fractions: fractions de profondeur à extraire (default: [0.25, 0.5, 0.75, 0.9])
            topk: nombre de tokens top-k à logger
            capture_logits: capturer les logits au dernier token
            capture_decomposition: capturer la décomposition attn/ffn (plus coûteux)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.topk = topk
        self.capture_logits = capture_logits
        self.capture_decomposition = capture_decomposition

        # Couches à extraire
        if layer_fractions is None:
            layer_fractions = [0.25, 0.5, 0.75, 0.9]
        self.target_layers = {
            f: max(0, min(num_layers - 1, int(f * num_layers)))
            for f in layer_fractions
        }
        logger.info(f"Target layers: {self.target_layers}")

        # Storage pour les captures
        self._captured: dict[int, torch.Tensor] = {}  # layer_idx → hidden states
        self._captured_attn: dict[int, torch.Tensor] = {}
        self._captured_ffn: dict[int, torch.Tensor] = {}
        self._hooks: list = []

        self._setup_hooks()

    def _setup_hooks(self):
        """Installe les forward hooks sur les couches cibles."""
        layers = self._get_layers()
        if layers is None:
            logger.warning("Cannot find model layers — signal extraction disabled")
            return

        for fraction, layer_idx in self.target_layers.items():
            if layer_idx < len(layers):
                target_layer = layers[layer_idx]

                # Hook sur la sortie de la couche complète
                hook = target_layer.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self._hooks.append(hook)

                # Hooks de décomposition attn/ffn (optionnel)
                if self.capture_decomposition:
                    if hasattr(target_layer, "self_attn"):
                        hook_attn = target_layer.self_attn.register_forward_hook(
                            self._make_attn_hook(layer_idx)
                        )
                        self._hooks.append(hook_attn)
                    if hasattr(target_layer, "mlp"):
                        hook_ffn = target_layer.mlp.register_forward_hook(
                            self._make_ffn_hook(layer_idx)
                        )
                        self._hooks.append(hook_ffn)

        logger.info(f"Installed {len(self._hooks)} hooks on {len(self.target_layers)} layers")

    def _get_layers(self):
        """Trouve la liste des couches du modèle."""
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "layers"):
                return inner.layers
            if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
                return inner.decoder.layers
        return None

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            elif hasattr(output, "last_hidden_state"):
                hidden = output.last_hidden_state
            else:
                hidden = output
            self._captured[layer_idx] = hidden.detach()
        return hook_fn

    def _make_attn_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._captured_attn[layer_idx] = output[0].detach()
            else:
                self._captured_attn[layer_idx] = output.detach()
        return hook_fn

    def _make_ffn_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._captured_ffn[layer_idx] = output[0].detach()
            else:
                self._captured_ffn[layer_idx] = output.detach()
        return hook_fn

    def extract_signals(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        output_logits: Optional[torch.Tensor] = None,
    ) -> StepSignals:
        """
        Extrait tous les signaux après un forward pass.
        
        Args:
            attention_mask: masque d'attention pour le mean pooling
            output_logits: logits de sortie du modèle (si disponibles)
            
        Returns:
            StepSignals avec tous les signaux
        """
        signals = StepSignals()

        # ── Hidden states à chaque couche cible ──
        for fraction, layer_idx in self.target_layers.items():
            if layer_idx not in self._captured:
                continue

            h = self._captured[layer_idx]  # (batch, seq_len, dim)

            # Mean pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(h.device, h.dtype)
                h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                h_mean = h.mean(dim=1)
            signals.hidden_states[f"layer{layer_idx}_mean"] = (
                h_mean.squeeze(0).cpu().float().numpy()
            )

            # Last token
            if attention_mask is not None:
                # Trouver le dernier token non-paddé
                seq_lengths = attention_mask.sum(dim=1) - 1
                last_idx = seq_lengths[0].item()
                h_last = h[0, int(last_idx), :]
            else:
                h_last = h[0, -1, :]
            signals.hidden_states[f"layer{layer_idx}_last"] = (
                h_last.cpu().float().numpy()
            )

            # Décomposition attn/ffn si disponible
            if self.capture_decomposition:
                if layer_idx in self._captured_attn:
                    h_attn = self._captured_attn[layer_idx]
                    signals.hidden_states[f"layer{layer_idx}_attn_last"] = (
                        h_attn[0, -1, :].cpu().float().numpy()
                    )
                if layer_idx in self._captured_ffn:
                    h_ffn = self._captured_ffn[layer_idx]
                    signals.hidden_states[f"layer{layer_idx}_ffn_last"] = (
                        h_ffn[0, -1, :].cpu().float().numpy()
                    )

        # ── Logits au dernier token ──
        if self.capture_logits and output_logits is not None:
            last_logits = output_logits[0, -1, :]  # (vocab_size,)

            # Entropie
            probs = F.softmax(last_logits, dim=-1)
            log_probs = F.log_softmax(last_logits, dim=-1)
            entropy = -(probs * log_probs).sum().item()
            signals.logits_entropy = entropy
            signals.logits_top1_prob = probs.max().item()

            # Top-k tokens et probabilités
            topk_probs, topk_indices = torch.topk(probs, min(self.topk, len(probs)))
            signals.logits_topk_probs = topk_probs.cpu().tolist()
            signals.logits_topk_tokens = [
                self.tokenizer.decode([idx.item()]).strip()
                for idx in topk_indices
            ]

        return signals

    def get_last_hidden_state(
        self, attention_mask: Optional[torch.Tensor] = None
    ) -> Optional[np.ndarray]:
        """
        Interface compatible avec HiddenStateLogger.
        Retourne le hidden state à depth 0.5, mean-poolé.
        """
        mid_layer = self.num_layers // 2
        # Trouver la couche la plus proche de 0.5
        closest = min(self.target_layers.values(), key=lambda l: abs(l - mid_layer))
        if closest in self._captured:
            h = self._captured[closest]
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(h.device, h.dtype)
                h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                h_mean = h.mean(dim=1)
            return h_mean.squeeze(0).cpu().float().numpy()
        return None

    def save_signals(self, task_id: str, step_idx: int, signals: StepSignals) -> str:
        """Sauvegarde tous les signaux d'un step."""
        step_dir = self.output_dir / f"{task_id}_step{step_idx:03d}"
        step_dir.mkdir(exist_ok=True)

        # Hidden states
        for key, h in signals.hidden_states.items():
            np.save(step_dir / f"{key}.npy", h)

        # Logits info
        logits_info = {
            "entropy": signals.logits_entropy,
            "top1_prob": signals.logits_top1_prob,
            "topk_tokens": signals.logits_topk_tokens[:20],
            "topk_probs": signals.logits_topk_probs[:20],
            "has_tool_call": signals.has_tool_call,
            "tool_name": signals.tool_name,
        }
        import json
        with open(step_dir / "logits.json", "w") as f:
            json.dump(logits_info, f, indent=2)

        return str(step_dir)

    def save(self, task_id: str, step_idx: int, h: np.ndarray) -> str:
        """Interface compatible avec HiddenStateLogger."""
        filename = f"{task_id}_step{step_idx:03d}.npy"
        path = self.output_dir / filename
        np.save(path, h)
        return str(path)

    def cleanup(self):
        """Retire tous les hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Analyses de la cascade Sprint 0C-bis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_layer_sweep(signals_dir: Path, labels: np.ndarray) -> dict:
    """
    Test 1b : AUC par couche et par mode de pooling.
    Produit la courbe AUC vs depth.
    """
    from sklearn.metrics import roc_auc_score

    results = {}
    step_dirs = sorted(signals_dir.glob("*_step*"))

    if not step_dirs:
        return {"error": "No signal directories found"}

    # Collecter les npy par clé
    keys = set()
    for sd in step_dirs[:1]:  # inspecter le premier
        for f in sd.glob("*.npy"):
            keys.add(f.stem)

    for key in sorted(keys):
        vectors = []
        valid_labels = []
        for i, sd in enumerate(step_dirs):
            npy_path = sd / f"{key}.npy"
            if npy_path.exists() and i < len(labels):
                vectors.append(np.load(npy_path))
                valid_labels.append(labels[i])

        if len(vectors) < 10 or len(set(valid_labels)) < 2:
            continue

        H = np.stack(vectors)
        y = np.array(valid_labels)

        # Direction mean-diff
        h_pos = H[y == 1]
        h_neg = H[y == 0]
        if len(h_pos) < 2 or len(h_neg) < 2:
            continue
        diff = h_pos.mean(axis=0) - h_neg.mean(axis=0)
        diff_norm = diff / (np.linalg.norm(diff) + 1e-12)

        projections = H @ diff_norm
        auc = roc_auc_score(y, projections)
        auc_inv = roc_auc_score(y, -projections)
        best_auc = max(auc, auc_inv)

        results[key] = {
            "auc": float(best_auc),
            "n_samples": len(vectors),
            "n_positive": int(y.sum()),
        }

    return results


def analyze_logits_structure(signals_dir: Path) -> dict:
    """
    Test 2b : structure de similarité entre tools depuis les logits top-k.
    Construit la matrice de confusion inter-tools.
    """
    import json
    from collections import defaultdict

    # Collecter les logits info pour les steps avec tool call
    tool_competitors = defaultdict(lambda: defaultdict(int))
    entropy_by_action = defaultdict(list)

    step_dirs = sorted(signals_dir.glob("*_step*"))
    for sd in step_dirs:
        logits_path = sd / "logits.json"
        if not logits_path.exists():
            continue
        with open(logits_path) as f:
            info = json.load(f)

        if not info.get("has_tool_call") or not info.get("tool_name"):
            continue

        tool = info["tool_name"]
        entropy_by_action[tool].append(info["entropy"])

        # Les top-k tokens comme concurrents
        for token, prob in zip(info.get("topk_tokens", [])[:10],
                               info.get("topk_probs", [])[:10]):
            if token != tool and prob > 0.01:
                tool_competitors[tool][token] += 1

    # Construire la matrice de confusion
    all_tools = sorted(tool_competitors.keys())
    confusion = {}
    for tool in all_tools:
        top_competitors = sorted(
            tool_competitors[tool].items(),
            key=lambda x: -x[1]
        )[:5]
        confusion[tool] = top_competitors

    # Entropie moyenne par tool
    entropy_stats = {
        tool: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }
        for tool, vals in entropy_by_action.items()
    }

    return {
        "tool_competitors": confusion,
        "entropy_by_tool": entropy_stats,
        "mean_entropy_tool_calls": float(np.mean([
            v for vals in entropy_by_action.values() for v in vals
        ])) if entropy_by_action else 0,
    }


def analyze_token_embeddings(model, tokenizer, tool_names: list[str]) -> dict:
    """
    Test 2a : structure fonctionnelle dans les embeddings de tokens de tools.
    """
    embeddings = model.get_input_embeddings()

    tool_vectors = {}
    for name in tool_names:
        # Tokeniser le nom du tool
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        if token_ids:
            # Prendre l'embedding du premier token (ou la moyenne si multi-token)
            vecs = [embeddings.weight[tid].detach().cpu().float().numpy() for tid in token_ids]
            tool_vectors[name] = np.mean(vecs, axis=0)

    if len(tool_vectors) < 3:
        return {"error": "Not enough tool embeddings found"}

    # Matrice de similarité cosinus
    names = sorted(tool_vectors.keys())
    n = len(names)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vi = tool_vectors[names[i]]
            vj = tool_vectors[names[j]]
            sim_matrix[i, j] = float(
                np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12)
            )

    # Cluster purity (silhouette) si on a les catégories
    from baselines.tools import TOOLS
    tool_categories = {t.name: t.category for t in TOOLS}

    categories = [tool_categories.get(n, "unknown") for n in names]
    unique_cats = list(set(categories))
    if len(unique_cats) > 1:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        cat_labels = le.fit_transform(categories)
        vectors_array = np.stack([tool_vectors[n] for n in names])
        silhouette = float(silhouette_score(vectors_array, cat_labels, metric="cosine"))
    else:
        silhouette = 0.0

    return {
        "tool_names": names,
        "similarity_matrix": sim_matrix.tolist(),
        "silhouette_by_category": silhouette,
        "n_tools_found": len(tool_vectors),
        "n_tools_requested": len(tool_names),
    }
