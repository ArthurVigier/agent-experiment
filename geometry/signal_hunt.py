"""
signal_hunt.py
Cascade de fallback pour trouver le signal d'agentivité.

Si l'extraction standard (hidden states, couche médiane, mean pooling) donne
un AUC < 0.65, cette cascade teste systématiquement d'autres sources de signal
avant de conclure que la structure n'existe pas.

Ordre de la cascade (du plus rapide/gratuit au plus coûteux) :
  1. Token embeddings des noms de tools (0 forward pass)
  2. Last-token vs mean pooling (re-analyse des .npy existants)
  3. Layer sweep (hooks multi-couche)
  4. Logits top-k au moment du choix (re-run partiel)
  5. Décomposition attention vs FFN (hooks séparés)

Convention : docstrings français, code anglais.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Fallback 1 — Token embeddings des noms de tools
# Coût : 0 forward pass, 5 secondes
# ═══════════════════════════════════════════════════════════════════════════

def analyze_tool_token_embeddings(model, tokenizer, output_dir: Path) -> dict:
    """
    Analyse la structure fonctionnelle dans la matrice d'embedding du vocabulaire.
    
    Si le modèle a des tokens pour les noms de tools (web_search, python_execute, etc.),
    la distance entre ces embeddings dans la matrice d'embedding pourrait déjà capturer
    la structure fonctionnelle — sans aucun forward pass.
    """
    import torch

    logger.info("─── Fallback 1 : Token embeddings des noms de tools ───")

    tool_names = [
        "web_search", "fetch_url", "wikipedia_lookup", "arxiv_search", "get_weather",
        "python_execute", "python_eval", "shell_command", "install_package", "create_plot",
        "read_file", "write_file", "list_directory", "csv_analyze", "json_query",
        "send_email", "send_slack_message", "schedule_meeting", "create_todo", "translate_text",
        "calculator", "unit_convert", "summarize_text", "get_current_datetime", "generate_id",
    ]

    tool_categories = {
        "web_search": "search", "fetch_url": "search", "wikipedia_lookup": "search",
        "arxiv_search": "search", "get_weather": "search",
        "python_execute": "code", "python_eval": "code", "shell_command": "code",
        "install_package": "code", "create_plot": "code",
        "read_file": "file", "write_file": "file", "list_directory": "file",
        "csv_analyze": "file", "json_query": "file",
        "send_email": "comm", "send_slack_message": "comm", "schedule_meeting": "comm",
        "create_todo": "comm", "translate_text": "comm",
        "calculator": "data", "unit_convert": "data", "summarize_text": "data",
        "get_current_datetime": "data", "generate_id": "data",
    }

    embed_matrix = model.get_input_embeddings().weight.detach().cpu().float()

    # Encoder chaque nom de tool et extraire l'embedding moyen de ses tokens
    tool_embeddings = {}
    for name in tool_names:
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        if token_ids:
            embeds = embed_matrix[token_ids]  # (n_tokens, dim)
            tool_embeddings[name] = embeds.mean(dim=0).numpy()

    if len(tool_embeddings) < 5:
        logger.warning("Pas assez de tools encodables")
        return {"status": "insufficient_tools"}

    # Construire la matrice de similarité cosinus
    names = list(tool_embeddings.keys())
    vecs = np.stack([tool_embeddings[n] for n in names])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs_norm = vecs / norms
    sim_matrix = vecs_norm @ vecs_norm.T

    # Mesurer si les outils du même cluster sont plus proches
    same_cat_sims = []
    diff_cat_sims = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = sim_matrix[i, j]
            if tool_categories.get(names[i]) == tool_categories.get(names[j]):
                same_cat_sims.append(sim)
            else:
                diff_cat_sims.append(sim)

    same_mean = float(np.mean(same_cat_sims)) if same_cat_sims else 0
    diff_mean = float(np.mean(diff_cat_sims)) if diff_cat_sims else 0
    separation = same_mean - diff_mean

    # Silhouette score
    from sklearn.metrics import silhouette_score
    labels = [tool_categories.get(n, "unknown") for n in names]
    unique_labels = set(labels)
    if len(unique_labels) >= 2:
        sil = float(silhouette_score(vecs_norm, labels))
    else:
        sil = 0.0

    logger.info(f"  Intra-cluster similarity: {same_mean:.3f}")
    logger.info(f"  Inter-cluster similarity: {diff_mean:.3f}")
    logger.info(f"  Separation: {separation:.3f}")
    logger.info(f"  Silhouette score: {sil:.3f}")

    result = {
        "same_category_sim": same_mean,
        "diff_category_sim": diff_mean,
        "separation": separation,
        "silhouette": sil,
        "n_tools_encoded": len(tool_embeddings),
        "embedding_dim": vecs.shape[1],
    }

    # Verdict
    if sil > 0.15:
        logger.info(f"  ✓ Structure fonctionnelle détectée dans les embeddings de vocabulaire")
        result["verdict"] = "structure_found"
    elif sil > 0.05:
        logger.info(f"  ~ Structure faible dans les embeddings")
        result["verdict"] = "weak_structure"
    else:
        logger.info(f"  ✗ Pas de structure dans les embeddings de vocabulaire")
        result["verdict"] = "no_structure"

    # Sauvegarder la matrice de similarité
    np.save(output_dir / "tool_embed_sim_matrix.npy", sim_matrix)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Fallback 2 — Last-token extraction (re-analyse des .npy existants)
# Coût : 0 forward pass (mais nécessite que le logger ait sauvé les states non-poolés)
# Si les .npy sont mean-poolés, il faut re-extraire → voir Fallback 3
# ═══════════════════════════════════════════════════════════════════════════

# Note : pour que ce fallback fonctionne sans re-run, il faudrait que le
# HiddenStateLogger sauvegarde le hidden state complet (batch, seq, dim)
# et pas seulement le mean-poolé. C'est plus lourd en stockage mais ça
# permet les re-analyses offline. En Sprint 0 on fait le compromis de
# sauvegarder le mean-poolé par défaut et de faire un re-run ciblé ici.


# ═══════════════════════════════════════════════════════════════════════════
# Fallback 2+3 combiné — Multi-signal extraction (nécessite le modèle en VRAM)
# Un seul forward pass qui capture tout : multi-layer, last-token, logits, attn/ffn
# ═══════════════════════════════════════════════════════════════════════════

class MultiSignalExtractor:
    """
    Extracteur multi-signal pour la cascade de fallback.
    
    Un seul forward pass capture simultanément :
    - Hidden states à chaque couche (pas seulement la médiane)
    - Last-token en plus du mean pooling
    - Logits top-k au dernier token
    - Décomposition attention vs FFN (si activé)
    
    Exécuté sur un subset de tâches (pas toutes les 60) pour limiter le compute.
    """

    def __init__(self, model, tokenizer, num_layers: int,
                 layer_stride: int = 4,
                 capture_logits: bool = True,
                 capture_attn_ffn: bool = False,
                 top_k_logits: int = 20,
                 device: str = "cuda"):
        """
        Args:
            model: le modèle HuggingFace
            tokenizer: le tokenizer
            num_layers: nombre total de couches
            layer_stride: extraire 1 couche sur N (4 = couches 0, 4, 8, ...)
            capture_logits: capturer les logits top-k
            capture_attn_ffn: décomposer attention et FFN séparément (plus coûteux)
            top_k_logits: nombre de tokens top-k à logger
        """
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.layer_stride = layer_stride
        self.capture_logits = capture_logits
        self.capture_attn_ffn = capture_attn_ffn
        self.top_k_logits = top_k_logits
        self.device = device

        # Couches à extraire
        self.target_layers = list(range(0, num_layers, layer_stride))
        if (num_layers - 1) not in self.target_layers:
            self.target_layers.append(num_layers - 1)

        # Stockage des captures
        self._captured_hidden = {}  # layer_idx → tensor
        self._hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Installe les hooks sur les couches cibles."""
        import torch

        layers = None
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "layers"):
                layers = inner.layers

        if layers is None:
            logger.warning("Cannot find model layers for multi-signal extraction")
            return

        for layer_idx in self.target_layers:
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self._hooks.append(hook)

        logger.info(f"Multi-signal hooks on layers: {self.target_layers}")

    def _make_hook(self, layer_idx: int):
        """Crée un hook pour une couche spécifique."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            elif hasattr(output, "last_hidden_state"):
                hidden = output.last_hidden_state
            else:
                hidden = output
            self._captured_hidden[layer_idx] = hidden.detach()
        return hook_fn

    def extract(self, text: str) -> dict:
        """
        Forward pass unique, extraction multi-signal.
        
        Returns:
            {
                "hidden_mean": {layer_idx: np.ndarray (dim,)},
                "hidden_last": {layer_idx: np.ndarray (dim,)},
                "logits_topk": [(token_str, prob), ...],
                "logits_entropy": float,
            }
        """
        import torch

        self._captured_hidden.clear()

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs)

        result = {
            "hidden_mean": {},
            "hidden_last": {},
            "seq_len": seq_len,
        }

        # Extraire mean-pool et last-token pour chaque couche capturée
        attn_mask = inputs.get("attention_mask")
        for layer_idx, h in self._captured_hidden.items():
            # h shape: (1, seq_len, dim)
            h_np = h.squeeze(0).cpu().float().numpy()

            # Mean pooling
            if attn_mask is not None:
                mask = attn_mask.squeeze(0).cpu().float().numpy()
                h_mean = (h_np * mask[:, None]).sum(axis=0) / (mask.sum() + 1e-12)
            else:
                h_mean = h_np.mean(axis=0)

            # Last token
            h_last = h_np[-1]

            result["hidden_mean"][layer_idx] = h_mean
            result["hidden_last"][layer_idx] = h_last

        # Logits top-k
        if self.capture_logits and hasattr(outputs, "logits"):
            logits = outputs.logits[0, -1]  # dernier token
            probs = torch.softmax(logits, dim=-1)

            # Entropie
            entropy = float(-torch.sum(probs * torch.log(probs + 1e-12)).cpu())
            result["logits_entropy"] = entropy

            # Top-k
            topk_vals, topk_ids = torch.topk(probs, self.top_k_logits)
            topk_tokens = []
            for val, idx in zip(topk_vals.cpu().numpy(), topk_ids.cpu().numpy()):
                token_str = self.tokenizer.decode([int(idx)])
                topk_tokens.append((token_str.strip(), float(val)))
            result["logits_topk"] = topk_tokens

        return result

    def cleanup(self):
        """Retire tous les hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def run_signal_hunt(
    model,
    tokenizer,
    num_layers: int,
    hidden_dim: int,
    traces: list,
    model_output_dir: Path,
    auc_threshold: float = 0.65,
    current_auc: float = 0.0,
    max_probe_tasks: int = 20,
    device: str = "cuda",
) -> dict:
    """
    Exécute la cascade de fallback complète.
    
    Appelé quand l'AUC standard est < auc_threshold.
    Teste chaque source de signal dans l'ordre et s'arrête dès
    qu'un signal fort est trouvé.
    
    Args:
        model: modèle en VRAM
        tokenizer: tokenizer
        num_layers: nombre de couches
        hidden_dim: dimension des hidden states
        traces: traces de Sprint 0A
        model_output_dir: répertoire de sortie
        auc_threshold: seuil en-dessous duquel on lance la cascade
        current_auc: AUC obtenue avec la méthode standard
        max_probe_tasks: nombre de tâches pour les re-extractions
        device: cuda ou cpu
    """
    from sklearn.metrics import roc_auc_score

    logger.info("\n" + "=" * 70)
    logger.info(f"═══ SIGNAL HUNT — AUC standard = {current_auc:.3f} < {auc_threshold} ═══")
    logger.info("Cascade de fallback pour localiser le signal d'agentivité")
    logger.info("=" * 70)

    hunt_results = {
        "trigger_auc": current_auc,
        "threshold": auc_threshold,
        "fallbacks_tested": [],
        "best_signal": None,
        "best_auc": current_auc,
        "best_config": "standard (depth=0.5, mean_pool)",
    }

    # ─── Fallback 1 : Token embeddings ───
    logger.info("\n[1/5] Token embeddings des noms de tools")
    fb1 = analyze_tool_token_embeddings(model, tokenizer, model_output_dir)
    hunt_results["fallbacks_tested"].append({"name": "token_embeddings", "result": fb1})

    # ─── Fallback 2+3+4 combiné : Multi-signal extraction ───
    # Sélectionner un subset de tâches avec tool calls
    probe_steps = _select_probe_steps(traces, max_probe_tasks)

    if not probe_steps:
        logger.warning("Pas assez de steps avec tool calls pour la cascade")
        _save_and_report(hunt_results, model_output_dir)
        return hunt_results

    logger.info(f"\nRe-extraction multi-signal sur {len(probe_steps)} steps")

    # Initialiser l'extracteur
    extractor = MultiSignalExtractor(
        model, tokenizer, num_layers,
        layer_stride=max(1, num_layers // 8),  # ~8 couches
        capture_logits=True,
        device=device,
    )

    # Extraire
    all_extractions = []
    labels = []
    for step_info in probe_steps:
        result = extractor.extract(step_info["text"])
        result["has_tool_call"] = step_info["has_tool_call"]
        result["tool_name"] = step_info.get("tool_name")
        all_extractions.append(result)
        labels.append(1 if step_info["has_tool_call"] else 0)

    extractor.cleanup()
    labels = np.array(labels)

    if len(set(labels)) < 2:
        logger.warning("Pas assez de diversité dans les labels")
        _save_and_report(hunt_results, model_output_dir)
        return hunt_results

    # ─── Fallback 2 : Last-token vs Mean pooling par couche ───
    logger.info("\n[2/5] Last-token vs mean pooling — layer sweep")

    layer_results = {}
    for layer_idx in extractor.target_layers:
        # Mean pool
        vecs_mean = np.stack([e["hidden_mean"][layer_idx] for e in all_extractions
                              if layer_idx in e["hidden_mean"]])
        # Last token
        vecs_last = np.stack([e["hidden_last"][layer_idx] for e in all_extractions
                              if layer_idx in e["hidden_last"]])

        if len(vecs_mean) != len(labels):
            continue

        for pool_name, vecs in [("mean", vecs_mean), ("last", vecs_last)]:
            # Direction moyenne tool vs non-tool
            h_tool = vecs[labels == 1]
            h_notool = vecs[labels == 0]
            if len(h_tool) < 2 or len(h_notool) < 2:
                continue

            diff = h_tool.mean(axis=0) - h_notool.mean(axis=0)
            diff_norm = diff / (np.linalg.norm(diff) + 1e-12)
            proj = vecs @ diff_norm

            auc = roc_auc_score(labels, proj)
            auc_inv = roc_auc_score(labels, -proj)
            best_auc = max(auc, auc_inv)

            depth = layer_idx / (num_layers - 1) if num_layers > 1 else 0
            key = f"layer_{layer_idx}_d{depth:.2f}_{pool_name}"
            layer_results[key] = {
                "layer": layer_idx,
                "depth": round(depth, 3),
                "pooling": pool_name,
                "auc": float(best_auc),
            }

            if best_auc > hunt_results["best_auc"]:
                hunt_results["best_auc"] = float(best_auc)
                hunt_results["best_config"] = f"layer={layer_idx} (depth={depth:.2f}), pooling={pool_name}"
                hunt_results["best_signal"] = key

    # Trier par AUC
    sorted_layers = sorted(layer_results.values(), key=lambda x: -x["auc"])
    logger.info(f"  Top 5 configurations :")
    for cfg in sorted_layers[:5]:
        marker = " ←" if cfg["auc"] > auc_threshold else ""
        logger.info(f"    layer={cfg['layer']:2d} (d={cfg['depth']:.2f}) "
                     f"{cfg['pooling']:4s}  AUC={cfg['auc']:.3f}{marker}")

    hunt_results["fallbacks_tested"].append({
        "name": "layer_sweep",
        "all_results": layer_results,
        "best": sorted_layers[0] if sorted_layers else None,
    })

    # ─── Fallback 3 : Logits — entropie et similarité inter-tools ───
    logger.info("\n[3/5] Logits — entropie et structure inter-tools")

    entropies_tool = []
    entropies_notool = []
    tool_call_logit_profiles = {}  # tool_name → list of topk distributions

    for ext, label in zip(all_extractions, labels):
        if "logits_entropy" in ext:
            if label == 1:
                entropies_tool.append(ext["logits_entropy"])
            else:
                entropies_notool.append(ext["logits_entropy"])

        if label == 1 and ext.get("tool_name") and "logits_topk" in ext:
            tn = ext["tool_name"]
            if tn not in tool_call_logit_profiles:
                tool_call_logit_profiles[tn] = []
            tool_call_logit_profiles[tn].append(ext["logits_topk"])

    logits_result = {}
    if entropies_tool and entropies_notool:
        ent_tool_mean = float(np.mean(entropies_tool))
        ent_notool_mean = float(np.mean(entropies_notool))
        logits_result["entropy_tool_mean"] = ent_tool_mean
        logits_result["entropy_notool_mean"] = ent_notool_mean
        logits_result["entropy_drop"] = ent_notool_mean - ent_tool_mean

        logger.info(f"  Entropie moyenne (tool call):    {ent_tool_mean:.2f}")
        logger.info(f"  Entropie moyenne (no tool call): {ent_notool_mean:.2f}")
        logger.info(f"  Drop d'entropie: {logits_result['entropy_drop']:.2f}")

        # AUC avec l'entropie comme prédicteur (entropie basse → tool call)
        all_entropies = np.array(entropies_tool + entropies_notool)
        ent_labels = np.array([1] * len(entropies_tool) + [0] * len(entropies_notool))
        auc_ent = roc_auc_score(ent_labels, -all_entropies)  # négatif car basse entropie = tool
        auc_ent_inv = roc_auc_score(ent_labels, all_entropies)
        logits_result["auc_entropy"] = float(max(auc_ent, auc_ent_inv))
        logger.info(f"  AUC(entropie → tool call): {logits_result['auc_entropy']:.3f}")

        if logits_result["auc_entropy"] > hunt_results["best_auc"]:
            hunt_results["best_auc"] = logits_result["auc_entropy"]
            hunt_results["best_config"] = "logits_entropy"
            hunt_results["best_signal"] = "logits_entropy"

    # Matrice de confusion inter-tools depuis les logits
    if tool_call_logit_profiles:
        logger.info(f"  Profils logits collectés pour {len(tool_call_logit_profiles)} tools")
        # Pour chaque tool, regarder quels autres tokens de tools apparaissent dans le top-k
        # C'est la matrice de "confusion" implicite du modèle
        logits_result["tool_profiles"] = {
            tn: len(profiles) for tn, profiles in tool_call_logit_profiles.items()
        }

    hunt_results["fallbacks_tested"].append({"name": "logits_analysis", "result": logits_result})

    # ─── Fallback 4 : Combinaison des signaux ───
    logger.info("\n[4/5] Combinaison des meilleurs signaux")

    # Si on a le best layer+pooling, construire un classifieur simple combiné
    if sorted_layers and sorted_layers[0]["auc"] > current_auc + 0.05:
        best_layer_cfg = sorted_layers[0]
        pool_key = "hidden_last" if best_layer_cfg["pooling"] == "last" else "hidden_mean"
        best_layer_idx = best_layer_cfg["layer"]

        vecs = np.stack([e[pool_key][best_layer_idx] for e in all_extractions
                         if best_layer_idx in e[pool_key]])

        if len(vecs) == len(labels):
            # Tester un classifieur linéaire simple (logistic regression)
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            clf = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(clf, vecs, labels, cv=min(5, len(labels) // 4), scoring="roc_auc")
            cv_auc = float(scores.mean())

            logger.info(f"  Logistic regression CV AUC (best layer): {cv_auc:.3f} ± {scores.std():.3f}")

            combo_result = {"cv_auc_best_layer": cv_auc, "cv_std": float(scores.std())}

            if cv_auc > hunt_results["best_auc"]:
                hunt_results["best_auc"] = cv_auc
                hunt_results["best_config"] = f"logistic_regression on {best_layer_cfg}"
                hunt_results["best_signal"] = "learned_probe"

            hunt_results["fallbacks_tested"].append({"name": "combined_probe", "result": combo_result})

    # ─── Fallback 5 : Verdict et recommandation ───
    logger.info("\n[5/5] Attn/FFN decomposition")
    logger.info("  → Skipped (nécessite des hooks custom sur les sous-modules)")
    logger.info("  → À implémenter si les fallbacks 1-4 sont tous négatifs")

    _save_and_report(hunt_results, model_output_dir)
    return hunt_results


def _select_probe_steps(traces: list, max_steps: int) -> list[dict]:
    """
    Sélectionne un subset équilibré de steps pour le re-run.
    50% avec tool call, 50% sans.
    """
    tool_steps = []
    notool_steps = []

    for trace in traces:
        steps = trace.steps if hasattr(trace, "steps") else []
        for step in steps:
            input_text = step.input_text if hasattr(step, "input_text") else ""
            if not input_text:
                # Reconstruire un texte approximatif
                input_text = step.generated_text if hasattr(step, "generated_text") else ""
            if not input_text:
                continue

            has_tool = step.action is not None if hasattr(step, "action") else False
            info = {
                "text": input_text,
                "has_tool_call": has_tool,
                "tool_name": step.action if has_tool else None,
                "task_id": trace.task_id if hasattr(trace, "task_id") else "",
            }

            if has_tool:
                tool_steps.append(info)
            else:
                notool_steps.append(info)

    # Équilibrer
    n_each = min(max_steps // 2, len(tool_steps), len(notool_steps))
    if n_each < 3:
        return []

    # Sélectionner aléatoirement (mais reproduciblement)
    rng = np.random.RandomState(42)
    selected_tool = [tool_steps[i] for i in rng.choice(len(tool_steps), n_each, replace=False)]
    selected_notool = [notool_steps[i] for i in rng.choice(len(notool_steps), n_each, replace=False)]

    return selected_tool + selected_notool


def _save_and_report(hunt_results: dict, output_dir: Path):
    """Sauvegarde et affiche le verdict final."""

    # Sauvegarder
    with open(output_dir / "signal_hunt.json", "w") as f:
        json.dump(hunt_results, f, indent=2, default=str)

    # Verdict
    best_auc = hunt_results["best_auc"]
    best_cfg = hunt_results["best_config"]
    trigger = hunt_results["trigger_auc"]

    print("\n" + "=" * 70)
    print("SIGNAL HUNT — VERDICT")
    print("=" * 70)
    print(f"  AUC standard (depth=0.5, mean pool): {trigger:.3f}")
    print(f"  Meilleur AUC trouvé:                 {best_auc:.3f}")
    print(f"  Configuration optimale:              {best_cfg}")
    print()

    improvement = best_auc - trigger
    if best_auc > 0.75:
        print(f"  ✓ SIGNAL TROUVÉ (AUC={best_auc:.3f}, gain={improvement:+.3f})")
        print(f"    → Utiliser la configuration '{best_cfg}' au lieu du standard")
        print(f"    → Mettre à jour le HiddenStateLogger pour extraire à cette config")
        print(f"    → Sprint 1 viable avec cette source de signal")
    elif best_auc > 0.65:
        print(f"  ~ SIGNAL PARTIEL (AUC={best_auc:.3f}, gain={improvement:+.3f})")
        print(f"    → Le signal existe mais est bruité")
        print(f"    → Un probe supervisé (logistic regression) pourrait suffire")
        print(f"    → Sprint 1B (probe) plus prometteur que Sprint 1A (PCA)")
    elif best_auc > trigger + 0.05:
        print(f"  ~ AMÉLIORATION MODESTE (AUC={best_auc:.3f}, gain={improvement:+.3f})")
        print(f"    → Le signal est réel mais faible partout")
        print(f"    → Considérer un embedding contrastif supervisé (Sprint 1A variante)")
        print(f"    → Le modèle n'a peut-être pas assez de structure à cette taille")
    else:
        print(f"  ✗ PAS DE SIGNAL EXPLOITABLE (meilleur AUC={best_auc:.3f})")
        print(f"    → Aucune source testée ne donne un signal > {trigger + 0.05:.3f}")
        print(f"    → Pivoter vers un embedding appris (contrastif supervisé)")
        print(f"    → Ou tester un modèle plus gros (le signal émerge peut-être à >3B)")
        print(f"    → Ou publier le résultat négatif : 'tool-use has no geometric")
        print(f"      structure in {best_cfg}' — c'est informatif")
    print("=" * 70)
