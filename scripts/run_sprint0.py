"""
run_sprint0.py
Script principal Sprint 0 — tourne sur RunPod (A100 80GB).

Exécute séquentiellement pour CHAQUE modèle :
  0-PRE : Structure fonctionnelle dans les embeddings de vocabulaire
  0A : Agent baseline sur toutes les tâches (avec logging hidden states)
  0B : Diagnostic d'échecs
  0C : Extraction Â sur les traces réelles

Puis en fin de run :
  0D : Analyse comparative cross-modèle (scaling + variantes)

Usage :
  # Un seul modèle
  python run_sprint0.py --model Qwen/Qwen3-8B

  # Scaling analysis (1.7B + 4B + 8B)
  python run_sprint0.py --scaling-preset

  # Variant analysis : base vs instruct vs abliterated (dissociation R̂ ↔ Â)
  python run_sprint0.py --variant-analysis

  # Full analysis : scaling + variants combinés
  python run_sprint0.py --full-analysis

  # Custom
  python run_sprint0.py --models Qwen/Qwen3-8B huihui-ai/Huihui-Qwen3-8B-abliterated-v2

Convention : docstrings français, code anglais.
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sprint0.log"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Modèles recommandés pour l'analyse de scaling ──────────────────────────
# Tous de la même famille (Qwen3-Instruct) pour isoler l'effet de la taille.
# Qwen3 (avril 2025+) remplace Qwen2.5 : meilleur tool calling natif,
# support MCP, Apache 2.0, et Qwen3-4B rivalise avec Qwen2.5-72B-Instruct.
#
# VRAM estimée en bf16 :
#   0.6B →  ~1.5 GB   (trivial)
#   1.7B →  ~4 GB     (trivial)
#   4B   →  ~8 GB     (trivial)
#   8B   → ~16 GB     (confortable sur A100 80GB)
#   14B  → ~28 GB     (confortable sur A100 80GB)
#   32B  → ~64 GB     (tight sur A100 80GB, OK en bf16)
#
# Le MoE Qwen3-30B-A3B est aussi intéressant : 3B activés, performance
# de 32B dense. Tient en ~8GB bf16. Bon point de comparaison efficience.
#
# Recommandation : 1.7B + 4B + 8B pour le meilleur rapport signal/compute.
# Optionnel : ajouter 0.6B (plancher) et/ou 30B-A3B (efficience MoE).

DEFAULT_MODELS = [
    "Qwen/Qwen3-8B",
]

SCALING_MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

# Variantes base/instruct/abliterated pour l'analyse de dissociation R̂ ↔ Â
# Base : pretraining pur, aucun alignement
# Instruct : fine-tuning RLHF/SFT, tool calling, refusal
# Abliterated : instruct avec R̂ chirurgicalement retirée
VARIANT_MODELS = [
    "Qwen/Qwen3-8B",                              # base (dense, pas instruct)
    "Qwen/Qwen3-8B-Instruct",                     # instruct (RLHF + tool calling)  -- NOTE: vérifier le nom exact sur HF
    "huihui-ai/Huihui-Qwen3-8B-abliterated-v2",   # abliterated (R̂ retirée)
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sprint 0 — Baseline + Diagnostic + Â + Scaling")

    # Modèles : --model (rétrocompatible) ou --models (multi)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, default=None,
                             help="HuggingFace model ID (single model)")
    model_group.add_argument("--models", type=str, nargs="+", default=None,
                             help="Multiple HuggingFace model IDs for scaling analysis")
    parser.add_argument("--scaling-preset", action="store_true",
                        help="Use preset scaling models: 1.7B + 4B + 8B")
    parser.add_argument("--variant-analysis", action="store_true",
                        help="Compare base/instruct/abliterated variants of 8B (dissociation R̂ ↔ Â)")
    parser.add_argument("--full-analysis", action="store_true",
                        help="Scaling preset + variant analysis combined")

    parser.add_argument("--tasks", type=str, default="all",
                        choices=["all", "single", "chain", "adversarial"],
                        help="Subset de tâches à exécuter")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Nombre max de steps par tâche")
    parser.add_argument("--output-dir", type=str, default="results/sprint0",
                        help="Répertoire de sortie racine")
    parser.add_argument("--skip-0c", action="store_true",
                        help="Skip l'extraction Â (Sprint 0C)")
    parser.add_argument("--r-hat-dir", type=str, default=None,
                        help="Répertoire contenant les R̂ pré-calculés (.npy), nommés par modèle")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda ou cpu)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Température de génération")

    args = parser.parse_args()

    # Résoudre la liste de modèles
    if args.full_analysis:
        # Scaling + variants : dédupliquer (Qwen3-8B apparaît dans les deux)
        seen = set()
        args.model_list = []
        for m in SCALING_MODELS + VARIANT_MODELS:
            if m not in seen:
                args.model_list.append(m)
                seen.add(m)
    elif args.variant_analysis:
        args.model_list = VARIANT_MODELS
    elif args.scaling_preset:
        args.model_list = SCALING_MODELS
    elif args.models:
        args.model_list = args.models
    elif args.model:
        args.model_list = [args.model]
    else:
        args.model_list = DEFAULT_MODELS

    return args


def model_short_name(model_id: str) -> str:
    """Qwen/Qwen3-8B → qwen3-8b"""
    return model_id.split("/")[-1].lower()


def estimate_vram_gb(model_id: str) -> float:
    """Estimation grossière de la VRAM nécessaire en bf16."""
    name = model_id.lower()
    for size, vram in [("0.6b", 1.5), ("1.7b", 4), ("0.5b", 1), ("1.5b", 3),
                       ("3b", 6), ("4b", 8), ("7b", 14), ("8b", 16),
                       ("14b", 28), ("30b-a3b", 8), ("32b", 64), ("72b", 144)]:
        if size in name:
            return vram
    return 16  # default


def get_available_vram_gb() -> float:
    """Retourne la VRAM disponible en GB."""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return total
    return 0


def load_model(model_id: str, device: str):
    """Charge le modèle et le tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Chargement du modèle : {model_id}")

    # Vérifier la VRAM
    vram_needed = estimate_vram_gb(model_id)
    vram_available = get_available_vram_gb()
    if vram_available > 0 and vram_needed > vram_available * 0.9:
        logger.warning(
            f"VRAM potentiellement insuffisante : {model_id} nécessite ~{vram_needed:.0f}GB, "
            f"disponible : {vram_available:.0f}GB. Tentative de chargement quand même."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Détecter le nombre de couches
    num_layers = None
    if hasattr(model, "config"):
        for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
            if hasattr(model.config, attr):
                num_layers = getattr(model.config, attr)
                break
    if num_layers is None:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            num_layers = len(model.model.layers)
        else:
            num_layers = 32
            logger.warning(f"Cannot detect num_layers, defaulting to {num_layers}")

    hidden_dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else 4096

    # Compter les paramètres
    num_params = sum(p.numel() for p in model.parameters())
    num_params_b = num_params / 1e9

    logger.info(f"Modèle chargé : {num_params_b:.2f}B params, {num_layers} couches, dim {hidden_dim}")

    return model, tokenizer, num_layers, hidden_dim, num_params_b


def unload_model(model):
    """Décharge proprement le modèle de la VRAM."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Modèle déchargé, VRAM libérée")


def get_task_list(args):
    """Retourne la liste de tâches selon le filtre."""
    from baselines.tasks import ALL_TASKS, SINGLE_TASKS, CHAIN_TASKS, ADVERSARIAL_TASKS
    if args.tasks == "single":
        return SINGLE_TASKS
    elif args.tasks == "chain":
        return CHAIN_TASKS
    elif args.tasks == "adversarial":
        return ADVERSARIAL_TASKS
    return ALL_TASKS


def run_0_pre(model, tokenizer, model_output_dir: Path) -> dict:
    """
    Sprint 0-PRE : analyse de la structure fonctionnelle dans les embeddings de vocabulaire.
    
    Aucun forward pass. Juste une lookup dans model.get_input_embeddings().
    Temps : < 30 secondes. Doit tourner AVANT tout le reste.
    
    Ce test répond à la question : est-ce que la structure fonctionnelle
    (similarité entre tools du même cluster) existe déjà dans les poids
    du modèle, avant même de lancer l'agent ?
    
    Si oui → l'espace fonctionnel est "gratuit", le routing géométrique
    peut se construire directement sur les embeddings de vocabulaire.
    Si non → il faudra construire l'espace fonctionnel à partir des
    hidden states ou des logits (Sprint 1).
    """
    from baselines.tools import TOOLS

    logger.info("═══ SPRINT 0-PRE : Embeddings de Vocabulaire ═══")

    embedding_matrix = model.get_input_embeddings()

    # ── Extraire les embeddings de chaque tool ──
    tool_vectors = {}
    tool_token_info = {}

    for tool in TOOLS:
        name = tool.name
        # Tokeniser le nom du tool
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        tokens_decoded = [tokenizer.decode([tid]) for tid in token_ids]

        # Embedding : moyenne si le nom est multi-token
        vecs = []
        for tid in token_ids:
            vec = embedding_matrix.weight[tid].detach().cpu().float().numpy()
            vecs.append(vec)

        if vecs:
            tool_vec = np.mean(vecs, axis=0)
            tool_vectors[name] = tool_vec
            tool_token_info[name] = {
                "token_ids": token_ids,
                "tokens": tokens_decoded,
                "n_tokens": len(token_ids),
                "category": tool.category,
            }

    logger.info(f"Embeddings extraits pour {len(tool_vectors)}/{len(TOOLS)} tools")

    if len(tool_vectors) < 5:
        logger.warning("Trop peu d'embeddings — skip analyse")
        return {"error": "insufficient_embeddings", "n_found": len(tool_vectors)}

    # ── Matrice de similarité cosinus ──
    names = sorted(tool_vectors.keys())
    n = len(names)
    V = np.stack([tool_vectors[name] for name in names])
    # Normaliser
    V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    sim_matrix = V_norm @ V_norm.T

    # ── Silhouette par catégorie fonctionnelle ──
    categories = [tool_token_info[name]["category"] for name in names]
    unique_cats = sorted(set(categories))

    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    cat_labels = le.fit_transform(categories)

    if len(unique_cats) > 1:
        silhouette_avg = float(silhouette_score(V_norm, cat_labels, metric="cosine"))
        silhouette_per_sample = silhouette_samples(V_norm, cat_labels, metric="cosine")

        # Silhouette par catégorie
        silhouette_by_cat = {}
        for cat in unique_cats:
            mask = np.array(categories) == cat
            if mask.sum() > 1:
                silhouette_by_cat[cat] = float(silhouette_per_sample[mask].mean())
    else:
        silhouette_avg = 0.0
        silhouette_by_cat = {}

    # ── Similarité intra-cluster vs inter-cluster ──
    intra_sims = []
    inter_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if categories[i] == categories[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0
    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0
    separation = intra_mean - inter_mean

    # ── Top paires les plus similaires et les plus dissimilaires ──
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((names[i], names[j], float(sim_matrix[i, j]),
                          categories[i] == categories[j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    top_similar = [(a, b, sim, same_cat) for a, b, sim, same_cat in pairs[:10]]
    top_dissimilar = [(a, b, sim, same_cat) for a, b, sim, same_cat in pairs[-10:]]

    # ── PCA pour visualisation et structure ──
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(5, n - 1))
    V_pca = pca.fit_transform(V_norm)

    # ── Rapport ──
    print("\n" + "=" * 70)
    print("SPRINT 0-PRE : STRUCTURE FONCTIONNELLE DANS LES EMBEDDINGS")
    print("=" * 70)

    print(f"\nTools analysés : {n}")
    print(f"Dimension des embeddings : {V.shape[1]}")

    print(f"\n─── Silhouette par catégorie (cosinus) ───")
    print(f"  Silhouette moyenne : {silhouette_avg:.3f}")
    for cat, sil in sorted(silhouette_by_cat.items(), key=lambda x: -x[1]):
        n_tools = sum(1 for c in categories if c == cat)
        bar = "█" * int(max(0, sil) * 30)
        print(f"  {cat:15s} ({n_tools} tools) : {sil:+.3f}  {bar}")

    print(f"\n─── Séparation intra/inter cluster ───")
    print(f"  Similarité intra-cluster (même catégorie) : {intra_mean:.4f}")
    print(f"  Similarité inter-cluster (catégories diff) : {inter_mean:.4f}")
    print(f"  Séparation (intra - inter)                : {separation:+.4f}")

    print(f"\n─── Top 10 paires les plus similaires ───")
    for a, b, sim, same_cat in top_similar:
        marker = "✓" if same_cat else "✗"
        print(f"  {marker} {a:20s} ↔ {b:20s}  cos={sim:.4f}")

    print(f"\n─── PCA variance expliquée ───")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var:.3f}")

    # Verdict
    print(f"\n─── VERDICT 0-PRE ───")
    if silhouette_avg > 0.3 and separation > 0.05:
        print(f"  ✓ STRUCTURE FONCTIONNELLE PRÉSENTE (silhouette={silhouette_avg:.3f})")
        print(f"    Les tools du même cluster sont proches dans l'espace d'embedding.")
        print(f"    → L'espace fonctionnel v0 peut être construit directement")
        print(f"      depuis les embeddings de vocabulaire, sans forward pass.")
        print(f"    → Le routing géométrique a une base solide.")
        verdict = "structure_found"
    elif silhouette_avg > 0.1:
        print(f"  ~ STRUCTURE PARTIELLE (silhouette={silhouette_avg:.3f})")
        print(f"    Les clusters existent mais sont bruités.")
        print(f"    → Utilisable comme initialisation, à raffiner avec les hidden states.")
        verdict = "partial_structure"
    else:
        print(f"  ✗ PAS DE STRUCTURE (silhouette={silhouette_avg:.3f})")
        print(f"    Les embeddings de vocabulaire ne capturent pas la structure fonctionnelle.")
        print(f"    → L'espace fonctionnel devra être appris (contrastif) ou extrait")
        print(f"      des hidden states.")
        verdict = "no_structure"

    print("=" * 70)

    # ── Sauvegarder ──
    results = {
        "verdict": verdict,
        "n_tools": n,
        "embedding_dim": int(V.shape[1]),
        "silhouette_avg": silhouette_avg,
        "silhouette_by_category": silhouette_by_cat,
        "intra_cluster_similarity": intra_mean,
        "inter_cluster_similarity": inter_mean,
        "separation": separation,
        "top_similar_pairs": [
            {"tool_a": a, "tool_b": b, "cosine": s, "same_category": sc}
            for a, b, s, sc in top_similar
        ],
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "tool_token_info": tool_token_info,
    }

    with open(model_output_dir / "embedding_structure.json", "w") as f:
        json.dump(results, f, indent=2)

    # Sauvegarder les embeddings pour usage ultérieur (routing géométrique)
    np.save(model_output_dir / "tool_embeddings.npy", V)
    np.save(model_output_dir / "tool_similarity_matrix.npy", sim_matrix)

    # Sauvegarder l'ordre des tools pour la matrice
    with open(model_output_dir / "tool_embedding_names.json", "w") as f:
        json.dump(names, f)

    logger.info(f"Résultats 0-PRE sauvegardés dans {model_output_dir}")

    return results


def run_0a(model, tokenizer, num_layers, model_output_dir: Path, args) -> list:
    """Sprint 0A : exécuter l'agent baseline sur toutes les tâches."""
    from baselines.react_agent import ReActAgent, HiddenStateLogger, Trace
    from baselines.tools import TOOL_MAP

    logger.info("═══ SPRINT 0A : Agent Baseline ═══")

    hs_dir = model_output_dir / "hidden_states"
    tasks = get_task_list(args)
    logger.info(f"Tâches : {len(tasks)} ({args.tasks})")

    hs_logger = HiddenStateLogger(model, num_layers, hs_dir)

    agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=TOOL_MAP,
        hs_logger=hs_logger,
        max_steps=args.max_steps,
        max_new_tokens=512,
        temperature=args.temperature,
        device=args.device,
    )

    traces = []
    for i, task in enumerate(tasks):
        logger.info(f"[{i+1}/{len(tasks)}] {task.id} : {task.prompt[:80]}...")
        try:
            trace = agent.run(
                task_id=task.id,
                task_prompt=task.prompt,
                task_category=task.category,
                expected_tools=task.expected_tools,
            )
            traces.append(trace)
            status = "✓" if trace.final_answer else "✗"
            logger.info(f"  {status} steps={len(trace.steps)}, tools={trace.tools_used}, "
                        f"time={trace.total_time_ms:.0f}ms")
        except Exception as e:
            logger.error(f"  ✗ Exception : {e}")
            trace = Trace(
                task_id=task.id,
                task_prompt=task.prompt,
                task_category=task.category,
                expected_tools=task.expected_tools,
                error=str(e),
            )
            traces.append(trace)

    # Sauvegarder les traces
    traces_data = [t.to_dict() for t in traces]
    with open(model_output_dir / "traces.json", "w") as f:
        json.dump(traces_data, f, indent=2)
    logger.info(f"Traces sauvegardées : {model_output_dir / 'traces.json'}")

    hs_logger.cleanup()
    return traces


def run_0b(traces: list, model_output_dir: Path, args) -> dict:
    """Sprint 0B : diagnostic d'échecs."""
    from baselines.failure_analysis import analyze_traces, print_report

    logger.info("\n═══ SPRINT 0B : Diagnostic d'Échecs ═══")

    tasks = get_task_list(args)
    analysis = analyze_traces(traces, tasks)
    print_report(analysis, traces)

    serializable = json.loads(json.dumps(analysis, default=str))
    with open(model_output_dir / "failure_analysis.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return analysis


def run_0c(traces: list, num_layers: int, hidden_dim: int,
           model_output_dir: Path, model_id: str, args) -> dict:
    """Sprint 0C : extraction Â sur les traces réelles."""
    from sklearn.metrics import roc_auc_score

    logger.info("\n═══ SPRINT 0C : Extraction Â sur Traces Réelles ═══")

    mid_layer = num_layers // 2

    # Chercher R̂ pour ce modèle
    r_hat = None
    if args.r_hat_dir:
        r_hat_dir = Path(args.r_hat_dir)
        short = model_short_name(model_id)
        candidates = [
            r_hat_dir / f"{short}.npy",
            r_hat_dir / f"{short}_r_hat.npy",
            r_hat_dir / f"r_hat_{short}.npy",
        ]
        for cand in candidates:
            if cand.exists():
                r_hat = np.load(cand)
                r_hat = r_hat / np.linalg.norm(r_hat)
                logger.info(f"R̂ chargé depuis {cand}, dim={r_hat.shape}")
                break
        if r_hat is None:
            logger.warning(f"Aucun R̂ trouvé pour {model_id} dans {r_hat_dir}")

    # Collecter les hidden states et labels
    projections_rhat = []
    labels = []
    all_hidden_states = []

    for trace_data in traces:
        steps = trace_data.steps if hasattr(trace_data, "steps") else []
        for step in steps:
            hs_path = step.hidden_state_path if hasattr(step, "hidden_state_path") else None
            if hs_path is None or not Path(hs_path).exists():
                continue

            h = np.load(hs_path)
            has_tool_call = step.action is not None if hasattr(step, "action") else False

            all_hidden_states.append(h)
            labels.append(1 if has_tool_call else 0)

            if r_hat is not None and h.shape[0] == r_hat.shape[0]:
                proj = float(np.dot(h, r_hat))
                projections_rhat.append(proj)

    labels = np.array(labels)

    if len(all_hidden_states) == 0:
        logger.warning("Pas de hidden states collectés — skip 0C")
        return {}

    logger.info(f"Collected {len(all_hidden_states)} steps "
                f"({labels.sum()} tool calls, {len(labels) - labels.sum()} non-tool)")

    results_0c = {
        "n_steps": len(all_hidden_states),
        "n_tool_calls": int(labels.sum()),
        "hidden_dim": int(all_hidden_states[0].shape[0]),
        "mid_layer": mid_layer,
    }

    # ── AUC avec R̂ si disponible ──
    if projections_rhat and len(set(labels)) > 1:
        projections_rhat = np.array(projections_rhat)
        auc = roc_auc_score(labels[:len(projections_rhat)], projections_rhat)
        auc_inv = roc_auc_score(labels[:len(projections_rhat)], -projections_rhat)
        best_auc = max(auc, auc_inv)
        sign = "+" if auc >= auc_inv else "-"

        proj_tool = projections_rhat[labels[:len(projections_rhat)] == 1]
        proj_notool = projections_rhat[labels[:len(projections_rhat)] == 0]

        logger.info(f"AUC(R̂ → tool call) = {best_auc:.3f} (sign: {sign})")
        logger.info(f"  Proj tool:    {proj_tool.mean():.4f} ± {proj_tool.std():.4f}")
        logger.info(f"  Proj no-tool: {proj_notool.mean():.4f} ± {proj_notool.std():.4f}")

        results_0c.update({
            "auc_r_hat": float(best_auc),
            "sign": sign,
            "proj_tool_mean": float(proj_tool.mean()),
            "proj_tool_std": float(proj_tool.std()),
            "proj_notool_mean": float(proj_notool.mean()),
            "proj_notool_std": float(proj_notool.std()),
            "separation_rhat": float(abs(proj_tool.mean() - proj_notool.mean())),
        })

    # ── PCA-based Â (toujours, même sans R̂) ──
    H = np.stack(all_hidden_states)
    h_tool = H[labels == 1]
    h_notool = H[labels == 0]

    if len(h_tool) >= 2 and len(h_notool) >= 2:
        # Direction moyenne tool - non-tool
        diff = h_tool.mean(axis=0) - h_notool.mean(axis=0)
        diff_norm = diff / (np.linalg.norm(diff) + 1e-12)

        proj_diff = H @ diff_norm
        auc_diff = roc_auc_score(labels, proj_diff)
        auc_diff_inv = roc_auc_score(labels, -proj_diff)
        auc_diff_best = max(auc_diff, auc_diff_inv)

        logger.info(f"AUC(mean-diff direction → tool call) = {auc_diff_best:.3f}")

        # Séparation sur cette direction
        proj_tool_diff = proj_diff[labels == 1]
        proj_notool_diff = proj_diff[labels == 0]

        # PCA sur les hidden states
        from sklearn.decomposition import PCA
        n_components = min(10, len(H) - 1, H.shape[1])
        pca = PCA(n_components=n_components)
        H_pca = pca.fit_transform(H)

        pc_aucs = []
        for pc_idx in range(min(5, H_pca.shape[1])):
            pc_proj = H_pca[:, pc_idx]
            pc_auc = roc_auc_score(labels, pc_proj)
            pc_auc_inv = roc_auc_score(labels, -pc_proj)
            pc_aucs.append(max(pc_auc, pc_auc_inv))

        logger.info(f"AUC par PC : {['%.3f' % a for a in pc_aucs]}")

        # Sauvegarder la direction Â extraite
        np.save(model_output_dir / "a_hat_extracted.npy", diff_norm)

        results_0c.update({
            "auc_mean_diff": float(auc_diff_best),
            "separation_mean_diff": float(abs(proj_tool_diff.mean() - proj_notool_diff.mean())),
            "pc_aucs": [float(a) for a in pc_aucs],
            "pca_explained_variance": pca.explained_variance_ratio_[:5].tolist(),
        })

        # Alignement Â ↔ R̂
        if r_hat is not None and diff_norm.shape[0] == r_hat.shape[0]:
            cos_alignment = float(np.dot(diff_norm, r_hat))
            results_0c["cos_a_hat_r_hat"] = cos_alignment
            logger.info(f"cos(Â_extracted, R̂) = {cos_alignment:.3f}")

    # Sauvegarder
    with open(model_output_dir / "a_hat_traces.json", "w") as f:
        json.dump(results_0c, f, indent=2)

    # Verdict
    auc_best = results_0c.get("auc_r_hat", results_0c.get("auc_mean_diff", 0))
    print("\n" + "-" * 60)
    if auc_best > 0.75:
        print(f"  VERDICT 0C : ✓ Signal d'agentivité fort (AUC={auc_best:.3f})")
    elif auc_best > 0.6:
        print(f"  VERDICT 0C : ~ Signal partiel (AUC={auc_best:.3f})")
    else:
        print(f"  VERDICT 0C : ✗ Signal faible (AUC={auc_best:.3f})")
    print("-" * 60)

    return results_0c


def run_0c_signal_hunt(model, tokenizer, num_layers: int, hidden_dim: int,
                       traces: list, results_0c: dict,
                       model_output_dir: Path, args) -> dict:
    """
    Sprint 0C+ : cascade de fallback si le signal standard est faible.
    
    Appelé automatiquement quand l'AUC de 0C est < 0.65.
    Teste systématiquement d'autres sources de signal avant de conclure.
    """
    from geometry.signal_hunt import run_signal_hunt

    auc_best = results_0c.get("auc_r_hat", results_0c.get("auc_mean_diff", 0))

    if auc_best >= 0.65:
        logger.info("AUC >= 0.65 — signal hunt non nécessaire")
        return {}

    logger.info(f"\nAUC = {auc_best:.3f} < 0.65 — lancement de la cascade de fallback")

    hunt_results = run_signal_hunt(
        model=model,
        tokenizer=tokenizer,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        traces=traces,
        model_output_dir=model_output_dir,
        auc_threshold=0.65,
        current_auc=auc_best,
        max_probe_tasks=min(30, len(traces)),
        device=args.device,
    )

    return hunt_results


# ═══════════════════════════════════════════════════════════════════════════
# Sprint 0D — Analyse comparative cross-modèle
# ═══════════════════════════════════════════════════════════════════════════

def run_0d(all_results: dict, output_dir: Path):
    """
    Sprint 0D : analyse de scaling géométrique.

    Compare les performances et les signaux géométriques à travers
    les tailles de modèle. Produit le tableau central de l'analyse.

    Hypothèse clé : si le signal géométrique (AUC Â, séparation)
    est comparable entre petits et grands modèles alors que les performances
    textuelles divergent → JEPA-Agent comblera le gap chez les petits modèles.
    """
    logger.info("\n" + "=" * 70)
    logger.info("═══ SPRINT 0D : Analyse de Scaling Géométrique ═══")
    logger.info("=" * 70)

    if len(all_results) < 2:
        logger.info("Un seul modèle — skip de l'analyse comparative")
        return

    # Construire les lignes
    rows = []
    for model_id, result in sorted(all_results.items(), key=lambda x: x[1]["num_params_b"]):
        row = {
            "model": model_short_name(model_id),
            "params_b": result["num_params_b"],
            "num_layers": result["num_layers"],
            "hidden_dim": result["hidden_dim"],
            "success_rate": result["analysis"]["success_rate"],
            "dominant_failure": result["analysis"]["dominant_failure"],
            "mean_steps": result["analysis"]["step_stats"]["mean_steps"],
            "mean_tool_calls": result["analysis"]["step_stats"]["mean_tool_calls"],
        }

        for comp in ["single", "chain", "adversarial"]:
            comp_data = result["analysis"]["by_complexity"].get(comp, {})
            row[f"success_{comp}"] = comp_data.get("success_rate", 0)

        if "0c" in result and result["0c"]:
            oc = result["0c"]
            row["auc_r_hat"] = oc.get("auc_r_hat")
            row["auc_mean_diff"] = oc.get("auc_mean_diff")
            row["cos_a_r"] = oc.get("cos_a_hat_r_hat")
            row["separation"] = oc.get("separation_rhat", oc.get("separation_mean_diff"))
            row["pc1_auc"] = oc.get("pc_aucs", [None])[0]
            row["pca_var_1"] = oc.get("pca_explained_variance", [None])[0]

        # Embedding structure (Sprint 0-PRE)
        if "0_pre" in result and result["0_pre"]:
            pre = result["0_pre"]
            row["embed_silhouette"] = pre.get("silhouette_avg")
            row["embed_separation"] = pre.get("separation")
            row["embed_verdict"] = pre.get("verdict")

        for mode in ["wrong_tool", "no_tool", "loop", "wrong_timing",
                     "premature_stop", "hallucinated_tool", "max_steps"]:
            fm = result["analysis"]["failure_modes"].get(mode, {})
            row[f"fail_{mode}"] = fm.get("rate", 0) if isinstance(fm, dict) else 0

        rows.append(row)

    # ── Afficher les tables ──

    print("\n" + "=" * 90)
    print("TABLE 1 — PERFORMANCE TEXTUELLE PAR TAILLE DE MODÈLE")
    print("=" * 90)
    header = f"{'Modèle':<25s} {'Params':>7s} {'Dim':>5s} {'Success':>8s} {'Single':>8s} {'Chain':>8s} {'Steps':>6s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<25s} {r['params_b']:>6.1f}B {r['hidden_dim']:>5d} "
              f"{r['success_rate']:>7.1%} {r.get('success_single', 0):>7.1%} "
              f"{r.get('success_chain', 0):>7.1%} {r['mean_steps']:>6.1f}")

    print(f"\n{'=' * 90}")
    print("TABLE 2 — SIGNAL GÉOMÉTRIQUE PAR TAILLE DE MODÈLE")
    print("=" * 90)
    header2 = f"{'Modèle':<25s} {'AUC R̂':>8s} {'AUC Â':>8s} {'cos(Â,R̂)':>9s} {'Sépar.':>8s} {'PC1 var':>8s}"
    print(header2)
    print("-" * len(header2))
    for r in rows:
        def fmt(v, f=".3f"):
            return f"{v:{f}}" if v is not None else "   n/a"
        print(f"{r['model']:<25s} {fmt(r.get('auc_r_hat')):>8s} "
              f"{fmt(r.get('auc_mean_diff')):>8s} {fmt(r.get('cos_a_r')):>9s} "
              f"{fmt(r.get('separation'), '.4f'):>8s} {fmt(r.get('pca_var_1')):>8s}")

    print(f"\n{'=' * 90}")
    print("TABLE 3 — MODES D'ÉCHEC PAR TAILLE (% des tâches)")
    print("=" * 90)
    header3 = f"{'Modèle':<25s} {'wrong_tool':>10s} {'no_tool':>8s} {'loop':>6s} {'halluc.':>8s} {'prem.stop':>10s} {'max_step':>9s}"
    print(header3)
    print("-" * len(header3))
    for r in rows:
        print(f"{r['model']:<25s} "
              f"{r.get('fail_wrong_tool', 0):>9.1%} "
              f"{r.get('fail_no_tool', 0):>7.1%} "
              f"{r.get('fail_loop', 0):>5.1%} "
              f"{r.get('fail_hallucinated_tool', 0):>7.1%} "
              f"{r.get('fail_premature_stop', 0):>9.1%} "
              f"{r.get('fail_max_steps', 0):>8.1%}")

    # ── Analyse quantitative ──

    smallest, largest = rows[0], rows[-1]
    perf_gap = largest["success_rate"] - smallest["success_rate"]

    print(f"\n{'=' * 90}")
    print("ANALYSE DE SCALING")
    print("=" * 90)

    print(f"\nGap textuel ({smallest['model']} → {largest['model']}): {perf_gap:+.1%}")
    print(f"  Ratio de taille : {largest['params_b'] / smallest['params_b']:.1f}x")

    # Gap géométrique
    for key, label in [("auc_mean_diff", "AUC Â (mean-diff)"), ("auc_r_hat", "AUC R̂")]:
        s_val = smallest.get(key)
        l_val = largest.get(key)
        if s_val is not None and l_val is not None:
            geo_gap = l_val - s_val
            print(f"  Gap {label}: {geo_gap:+.3f} ({s_val:.3f} → {l_val:.3f})")

    # Taux d'échecs fixables par géométrie par modèle
    geo_fixable_modes = ["wrong_tool", "no_tool", "loop", "hallucinated_tool"]
    print(f"\n  Échecs corrigibles par JEPA-Agent (wrong_tool + no_tool + loop + halluc.):")
    for r in rows:
        geo_fixable = sum(r.get(f"fail_{m}", 0) for m in geo_fixable_modes)
        print(f"    {r['model']:<25s}: {geo_fixable:.1%}")

    # ── Verdict ──

    print(f"\n{'─' * 90}")
    print("VERDICT SCALING")
    print("─" * 90)

    geo_small = smallest.get("auc_mean_diff") or smallest.get("auc_r_hat") or 0
    geo_large = largest.get("auc_mean_diff") or largest.get("auc_r_hat") or 0
    geo_gap = abs(geo_large - geo_small)

    fixable_small = sum(smallest.get(f"fail_{m}", 0) for m in geo_fixable_modes)
    fixable_large = sum(largest.get(f"fail_{m}", 0) for m in geo_fixable_modes)

    if perf_gap > 0.15 and geo_small > 0.65 and geo_gap < 0.15:
        print("  ✓ COMPRESSION HYPOTHESIS SUPPORTED")
        print(f"    Signal géométrique comparable ({geo_small:.3f} vs {geo_large:.3f}, Δ={geo_gap:.3f})")
        print(f"    Performance textuelle divergente ({smallest['success_rate']:.1%} vs {largest['success_rate']:.1%})")
        print(f"    Échecs fixables : {fixable_small:.1%} (petit) vs {fixable_large:.1%} (grand)")
        print(f"    → Le {smallest['model']} a le signal mais pas le décodeur textuel")
        print(f"    → JEPA-Agent prédit un gain différentiel plus fort chez le petit modèle")
        print(f"    → RÉSULTAT PUBLIABLE : 'Geometric routing compresses the scaling curve'")
    elif perf_gap > 0.10 and geo_small > 0.60:
        print("  ~ PARTIAL SUPPORT")
        print(f"    Signal présent mais plus faible chez le petit modèle")
        print(f"    → JEPA-Agent aidera, gain différentiel probable mais modéré")
    elif geo_small < 0.55:
        print("  ✗ INSUFFICIENT SIGNAL AT SMALL SCALE")
        print(f"    AUC du petit modèle trop basse ({geo_small:.3f})")
        print(f"    → La structure d'agentivité n'émerge pas à {smallest['params_b']:.1f}B params")
        if len(rows) > 2:
            # Trouver le seuil
            for r in rows[1:]:
                r_auc = r.get("auc_mean_diff") or r.get("auc_r_hat") or 0
                if r_auc > 0.65:
                    print(f"    → Seuil d'émergence probable : ~{r['params_b']:.1f}B ({r['model']})")
                    break
    else:
        print(f"  ? INCONCLUSIVE")
        print(f"    Gap textuel faible ({perf_gap:.1%}) ou signal ambigu")
        print(f"    → Ajouter un modèle plus petit ou plus grand pour clarifier")

    print("─" * 90)

    # Sauvegarder
    scaling_analysis = {
        "models": rows,
        "perf_gap_textual": float(perf_gap),
        "geo_gap": float(geo_gap),
        "geo_small": float(geo_small),
        "geo_large": float(geo_large),
        "fixable_small": float(fixable_small),
        "fixable_large": float(fixable_large),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "scaling_analysis.json", "w") as f:
        json.dump(scaling_analysis, f, indent=2, default=str)
    logger.info(f"Analyse de scaling sauvegardée : {output_dir / 'scaling_analysis.json'}")

    # ── Analyse variantes base/instruct/abliterated ──
    _run_variant_analysis(rows, all_results, output_dir)


def _run_variant_analysis(rows: list, all_results: dict, output_dir: Path):
    """
    Analyse de dissociation R̂ ↔ Â via le triplet base/instruct/abliterated.
    
    Détecte automatiquement si les trois variantes sont présentes dans les résultats.
    Si oui, produit le tableau comparatif et le verdict de dissociation.
    """
    # Détecter les triplets : chercher des modèles qui partagent la même taille
    # mais diffèrent par base/instruct/abliterated
    model_names = [r["model"] for r in rows]

    # Heuristique : chercher "abliterated" et le modèle de base correspondant
    abliterated = [r for r in rows if "abliterat" in r["model"]]
    if not abliterated:
        return  # Pas de variante abliterated → skip

    # Trouver les variantes correspondantes (même taille approximative)
    base_candidates = [r for r in rows if "instruct" not in r["model"]
                       and "abliterat" not in r["model"]]
    instruct_candidates = [r for r in rows if "instruct" in r["model"]
                           or (r not in base_candidates and r not in abliterated)]

    # Matcher par taille de paramètres (~même nombre de params)
    for abl in abliterated:
        abl_params = abl["params_b"]
        base = None
        instruct = None

        for b in base_candidates:
            if abs(b["params_b"] - abl_params) / abl_params < 0.2:
                base = b
                break
        # Le modèle instruct peut avoir le même nombre de params (fine-tune du base)
        for inst in instruct_candidates:
            if abs(inst["params_b"] - abl_params) / abl_params < 0.2 and inst != base:
                instruct = inst
                break
        # Si on n'a pas trouvé d'instruct séparé, le base pourrait être l'instruct
        # (Qwen3-8B est le base, pas de variante instruct séparée dans Qwen3)
        # Dans ce cas on a base + abliterated seulement

        if not base and not instruct:
            continue

        print(f"\n{'=' * 90}")
        print("ANALYSE VARIANTES — DISSOCIATION R̂ ↔ Â")
        print("=" * 90)

        variants = {}
        if base:
            variants["base"] = base
        if instruct:
            variants["instruct"] = instruct
        variants["abliterated"] = abl

        # Table comparative
        header = f"{'Variante':<30s} {'Params':>7s} {'Success':>8s} {'AUC Â':>8s} {'AUC R̂':>8s} {'Embed sil.':>10s}"
        print(header)
        print("-" * len(header))
        for label, r in variants.items():
            def fmt(v, f=".3f"):
                return f"{v:{f}}" if v is not None else "   n/a"
            print(f"{label + ' (' + r['model'] + ')':<30s} "
                  f"{r['params_b']:>6.1f}B "
                  f"{r['success_rate']:>7.1%} "
                  f"{fmt(r.get('auc_mean_diff')):>8s} "
                  f"{fmt(r.get('auc_r_hat')):>8s} "
                  f"{fmt(r.get('embed_silhouette')):>10s}")

        # Verdict de dissociation
        print(f"\n{'─' * 90}")
        print("VERDICT DISSOCIATION R̂ ↔ Â")
        print("─" * 90)

        abl_auc = abl.get("auc_mean_diff") or abl.get("auc_r_hat") or 0
        ref_model = instruct if instruct else base
        ref_auc = ref_model.get("auc_mean_diff") or ref_model.get("auc_r_hat") or 0 if ref_model else 0
        ref_label = "instruct" if instruct else "base"
        base_auc = base.get("auc_mean_diff") or base.get("auc_r_hat") or 0 if base else 0

        if abl_auc > 0 and ref_auc > 0:
            delta = abl_auc - ref_auc

            if abs(delta) < 0.10 and abl_auc > 0.60:
                print(f"  ✓ SCÉNARIO A : Â SURVIT À L'ABLITÉRATION")
                print(f"    AUC {ref_label}={ref_auc:.3f}, abliterated={abl_auc:.3f} (Δ={delta:+.3f})")
                print(f"    → Â et R̂ sont des structures DISTINCTES")
                print(f"    → L'agentivité ne vit pas dans la direction de refusal")
                print(f"    → Â est un signal robuste, indépendant du mécanisme RLHF")
                print(f"    → RÉSULTAT PUBLIABLE : 'Agency direction is disentangled from refusal'")
            elif delta < -0.15:
                print(f"  ✗ SCÉNARIO B : Â DISPARAÎT AVEC L'ABLITÉRATION")
                print(f"    AUC {ref_label}={ref_auc:.3f}, abliterated={abl_auc:.3f} (Δ={delta:+.3f})")
                print(f"    → Â VIT dans la direction de refusal")
                print(f"    → Le routing géométrique marche toujours (on ne retire pas R̂)")
                print(f"    → Mais la thèse d'indépendance Â/R̂ est infirmée")
            elif delta > 0.10:
                print(f"  ★ SCÉNARIO C : Â EST PLUS FORT APRÈS ABLITÉRATION")
                print(f"    AUC {ref_label}={ref_auc:.3f}, abliterated={abl_auc:.3f} (Δ={delta:+.3f})")
                print(f"    → R̂ MASQUAIT le signal d'agentivité !")
                print(f"    → Le RLHF introduit une interférence refusal ↔ agentivité")
                print(f"    → Les modèles abliterated sont de MEILLEURES bases pour JEPA-Agent")
                print(f"    → RÉSULTAT TRÈS PUBLIABLE : 'RLHF refusal interferes with agency signal'")
            else:
                print(f"  ? RÉSULTAT AMBIGU")
                print(f"    AUC {ref_label}={ref_auc:.3f}, abliterated={abl_auc:.3f} (Δ={delta:+.3f})")
                print(f"    → Pas de conclusion claire, possible bruit ou effet marginal")

        # Signal dans le base (pretraining pur)
        if base and base_auc > 0:
            print(f"\n  Signal dans le base (pretraining pur) :")
            if base_auc > 0.65:
                print(f"    ✓ AUC base = {base_auc:.3f} — l'agentivité ÉMERGE du pretraining")
                print(f"    → Pas besoin de fine-tuning pour la structure géométrique")
                print(f"    → Renforce la thèse d'invariant de pretraining (comme R̂)")
            elif base_auc > 0.55:
                print(f"    ~ AUC base = {base_auc:.3f} — signal partiel dans le pretraining")
                print(f"    → Le fine-tuning amplifie une structure pré-existante")
            else:
                print(f"    ✗ AUC base = {base_auc:.3f} — pas de signal dans le pretraining")
                print(f"    → La structure d'agentivité est CRÉÉE par le fine-tuning")

        # Performance tool-use de l'abliterated
        if instruct and abl.get("success_rate") is not None:
            inst_sr = instruct["success_rate"]
            abl_sr = abl["success_rate"]
            sr_delta = abl_sr - inst_sr
            print(f"\n  Impact de l'ablitération sur le tool-use :")
            print(f"    Success rate instruct={inst_sr:.1%}, abliterated={abl_sr:.1%} (Δ={sr_delta:+.1%})")
            if abs(sr_delta) < 0.05:
                print(f"    → R̂ n'est PAS nécessaire pour le tool calling")
            elif sr_delta < -0.10:
                print(f"    → R̂ CONTRIBUE au tool calling (sa suppression dégrade)")
            else:
                print(f"    → L'ablitération AMÉLIORE le tool calling (!)")

        print("─" * 90)

        # Sauvegarder
        variant_analysis = {
            "variants": {label: r for label, r in variants.items()},
            "abl_auc": abl_auc,
            "ref_auc": ref_auc,
            "base_auc": base_auc,
            "delta_auc": abl_auc - ref_auc if ref_auc else None,
        }
        with open(output_dir / "variant_analysis.json", "w") as f:
            json.dump(variant_analysis, f, indent=2, default=str)
        logger.info(f"Analyse variantes sauvegardée : {output_dir / 'variant_analysis.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("SPRINT 0 — JEPA-Agent Baseline + Diagnostic + Â + Scaling")
    logger.info("=" * 70)
    logger.info(f"Modèles ({len(args.model_list)}) : {args.model_list}")
    logger.info(f"Device : {args.device}")
    if torch.cuda.is_available():
        logger.info(f"GPU : {torch.cuda.get_device_name(0)}, "
                     f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Output : {args.output_dir}")

    all_results = {}

    for model_idx, model_id in enumerate(args.model_list):
        logger.info("\n" + "╔" + "═" * 68 + "╗")
        logger.info(f"║ MODÈLE {model_idx + 1}/{len(args.model_list)} : {model_id}")
        logger.info("╚" + "═" * 68 + "╝")

        short_name = model_short_name(model_id)
        model_output_dir = output_dir / short_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Charger
        model, tokenizer, num_layers, hidden_dim, num_params_b = load_model(
            model_id, args.device
        )

        model_result = {
            "model_id": model_id,
            "short_name": short_name,
            "num_params_b": num_params_b,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
        }

        # ── Sprint 0-PRE : Structure fonctionnelle dans les embeddings de vocabulaire ──
        # Pas de forward pass. Juste une lookup dans la matrice d'embedding.
        # Donne un signal immédiat sur l'existence d'un espace fonctionnel.
        t0 = time.perf_counter()
        results_pre = run_0_pre(model, tokenizer, model_output_dir)
        t_pre = time.perf_counter() - t0
        logger.info(f"Sprint 0-PRE terminé en {t_pre:.1f}s")
        model_result["0_pre"] = results_pre

        # Sprint 0A
        t0 = time.perf_counter()
        traces = run_0a(model, tokenizer, num_layers, model_output_dir, args)
        t_0a = time.perf_counter() - t0
        logger.info(f"Sprint 0A terminé en {t_0a:.0f}s")

        # Sprint 0B
        t0 = time.perf_counter()
        analysis = run_0b(traces, model_output_dir, args)
        t_0b = time.perf_counter() - t0
        logger.info(f"Sprint 0B terminé en {t_0b:.0f}s")
        model_result["analysis"] = analysis

        # Sprint 0C
        t_0c = 0
        if not args.skip_0c:
            t0 = time.perf_counter()
            results_0c = run_0c(traces, num_layers, hidden_dim,
                                model_output_dir, model_id, args)
            t_0c = time.perf_counter() - t0
            logger.info(f"Sprint 0C terminé en {t_0c:.0f}s")
            model_result["0c"] = results_0c

            # Signal hunt si AUC faible
            hunt_results = run_0c_signal_hunt(
                model, tokenizer, num_layers, hidden_dim,
                traces, results_0c, model_output_dir, args
            )
            if hunt_results:
                model_result["signal_hunt"] = hunt_results
                # Mettre à jour le best AUC si la cascade a trouvé mieux
                if hunt_results.get("best_auc", 0) > results_0c.get("auc_mean_diff", 0):
                    results_0c["best_auc_after_hunt"] = hunt_results["best_auc"]
                    results_0c["best_config_after_hunt"] = hunt_results["best_config"]
        else:
            model_result["0c"] = {}

        model_result["timing"] = {
            "0_pre_seconds": t_pre,
            "0a_seconds": t_0a,
            "0b_seconds": t_0b,
            "0c_seconds": t_0c,
        }

        with open(model_output_dir / "summary.json", "w") as f:
            json.dump(model_result, f, indent=2, default=str)

        all_results[model_id] = model_result

        # Décharger avant le suivant
        unload_model(model)
        del tokenizer
        logger.info(f"Modèle {short_name} traité")

    # Sprint 0D : analyse comparative
    run_0d(all_results, output_dir)

    # Résumé final
    logger.info("\n" + "=" * 70)
    logger.info("SPRINT 0 COMPLET")
    logger.info("=" * 70)
    for model_id, result in sorted(all_results.items(),
                                    key=lambda x: x[1]["num_params_b"]):
        sr = result["analysis"]["success_rate"]
        df = result["analysis"]["dominant_failure"]
        auc = result.get("0c", {}).get("auc_mean_diff",
              result.get("0c", {}).get("auc_r_hat", "n/a"))
        pre = result.get("0_pre", {}).get("verdict", "n/a")
        sil = result.get("0_pre", {}).get("silhouette_avg", "n/a")
        sil_str = f"{sil:.3f}" if isinstance(sil, float) else sil
        logger.info(f"  {model_short_name(model_id):30s}  "
                     f"success={sr:.1%}  fail={df}  AUC_Â={auc}  "
                     f"embed={pre}(sil={sil_str})")
    logger.info(f"\nOutputs dans : {output_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
