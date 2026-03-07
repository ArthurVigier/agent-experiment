"""
evaluate_deep.py
Évaluation approfondie post-hoc des traces Sprint 0.

Se lance APRÈS le run principal, sur les données sauvegardées.
Utilise un LLM local comme juge pour évaluer la qualité des réponses,
ré-analyser les hidden states, et produire des métriques plus fiables
que la classification heuristique de failure_analysis.py.

Usage :
  # Évaluation complète avec LLM judge (utilise le même modèle ou un autre)
  python evaluate_deep.py --traces-dir results/sprint0_test/qwen3-8b

  # Avec un modèle juge différent (plus petit, plus rapide)
  python evaluate_deep.py --traces-dir results/sprint0_test/qwen3-8b --judge-model Qwen/Qwen3-4B

  # Sans LLM judge (seulement les métriques géométriques et heuristiques améliorées)
  python evaluate_deep.py --traces-dir results/sprint0_test/qwen3-8b --skip-judge

  # Évaluer plusieurs modèles et comparer
  python evaluate_deep.py --traces-dirs results/sprint0_test/qwen3-8b results/sprint0_test/qwen3-4b

Convention : docstrings français, code anglais.
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Chargement des données
# ═══════════════════════════════════════════════════════════════════════════

def load_traces(traces_dir: Path) -> tuple[list[dict], dict]:
    """Charge les traces et les métadonnées."""
    traces_path = traces_dir / "traces.json"
    if not traces_path.exists():
        raise FileNotFoundError(f"traces.json not found in {traces_dir}")

    with open(traces_path) as f:
        traces = json.load(f)

    # Charger les résultats existants si disponibles
    existing = {}
    for name in ["failure_analysis.json", "embedding_structure.json",
                 "a_hat_traces.json", "summary.json"]:
        path = traces_dir / name
        if path.exists():
            with open(path) as f:
                existing[name.replace(".json", "")] = json.load(f)

    logger.info(f"Loaded {len(traces)} traces from {traces_dir}")
    return traces, existing


# ═══════════════════════════════════════════════════════════════════════════
# 2. LLM-as-a-Judge
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI agent tool-use traces. 
You will be given a task prompt and the agent's execution trace (thoughts, actions, observations, final answer).
You must evaluate the trace on multiple criteria and respond ONLY with a JSON object.

Evaluation criteria:
1. task_completed (bool): Did the agent actually accomplish what was asked?
2. correct_tool_selection (0-3): 0=wrong tools, 1=partially right, 2=right but suboptimal, 3=optimal
3. efficiency (0-3): 0=wasteful/looping, 1=too many steps, 2=reasonable, 3=optimal path
4. answer_quality (0-3): 0=wrong/missing, 1=partial, 2=correct but incomplete, 3=complete and accurate
5. tool_usage_needed (bool): Did this task actually REQUIRE a tool, or could it be answered from knowledge?
6. failure_mode (string): One of: "success", "wrong_tool", "no_tool_when_needed", "unnecessary_tool_use", "loop", "wrong_params", "hallucinated_tool", "premature_stop", "partial_success", "correct_no_tool"
7. explanation (string): Brief explanation of your evaluation (1-2 sentences)

Respond ONLY with valid JSON, no markdown, no preamble."""

JUDGE_USER_TEMPLATE = """Task: {task_prompt}
Expected tools: {expected_tools}
Task category: {task_category}

Agent trace:
{trace_text}

Final answer: {final_answer}

Evaluate this trace. Respond with JSON only."""


def format_trace_for_judge(trace: dict) -> str:
    """Formate une trace pour le prompt du juge."""
    lines = []
    for step in trace.get("steps", []):
        thought = step.get("thought", "")
        action = step.get("action", "")
        params = step.get("action_params", {})
        observation = step.get("observation", "")

        if thought:
            lines.append(f"Thought: {thought[:200]}")
        if action:
            params_str = ", ".join(f'{k}="{v}"' for k, v in (params or {}).items())
            lines.append(f"Action: {action}({params_str})")
        if observation:
            lines.append(f"Observation: {observation[:300]}")
        lines.append("")

    return "\n".join(lines) if lines else "[No steps recorded]"


def run_judge_batch(
    traces: list[dict],
    model_id: str = "Qwen/Qwen3-4B",
    device: str = "cuda",
    batch_pause: float = 0.0,
) -> list[dict]:
    """
    Évalue chaque trace avec un LLM juge local.
    
    Utilise un modèle open-source (par défaut Qwen3-4B, plus léger
    que le 8B) pour juger les traces du 8B. On peut aussi utiliser
    le même modèle.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading judge model: {model_id}")
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

    judgments = []
    for i, trace in enumerate(traces):
        task_id = trace.get("task_id", f"task_{i}")
        logger.info(f"  Judging [{i+1}/{len(traces)}] {task_id}...")

        # Construire le prompt
        trace_text = format_trace_for_judge(trace)
        user_content = JUDGE_USER_TEMPLATE.format(
            task_prompt=trace.get("task_prompt", ""),
            expected_tools=", ".join(trace.get("expected_tools", [])) or "none specified",
            task_category=trace.get("task_category", ""),
            trace_text=trace_text[:2000],  # tronquer si trop long
            final_answer=trace.get("final_answer", "[No final answer]")[:500],
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Générer
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"System: {JUDGE_SYSTEM_PROMPT}\n\nUser: {user_content}\n\nAssistant:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()

            # Parser le JSON
            judgment = _parse_judge_response(response)
            judgment["task_id"] = task_id
            judgment["raw_response"] = response[:500]
            judgments.append(judgment)

        except Exception as e:
            logger.warning(f"  Judge error on {task_id}: {e}")
            judgments.append({
                "task_id": task_id,
                "error": str(e),
                "task_completed": None,
                "failure_mode": "judge_error",
            })

        if batch_pause > 0:
            time.sleep(batch_pause)

    # Cleanup
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Judged {len(judgments)} traces")
    return judgments


def _parse_judge_response(response: str) -> dict:
    """Parse la réponse JSON du juge, avec fallback."""
    # Nettoyer
    text = response.strip()

    # Retirer le bloc <think>...</think> si présent (Qwen3 thinking mode)
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Retirer les markdown fences
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Essayer de trouver un objet JSON dans le texte
    # Chercher le premier { ... } complet
    brace_depth = 0
    json_start = None
    for i, c in enumerate(text):
        if c == '{':
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
            if brace_depth == 0 and json_start is not None:
                try:
                    return json.loads(text[json_start:i + 1])
                except json.JSONDecodeError:
                    pass

    # Fallback : essayer le texte entier
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "parse_error": True,
            "raw": text[:300],
            "task_completed": None,
            "failure_mode": "judge_parse_error",
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Analyse des hidden states (post-hoc)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_hidden_states(traces_dir: Path, traces: list[dict]) -> dict:
    """
    Analyse post-hoc des hidden states sauvegardés.
    
    Métriques :
    - Séparation tool-call vs non-tool-call (à toutes les couches disponibles)
    - Clustering des hidden states par tool utilisé
    - Trajectoire dans l'espace latent au cours d'une tâche
    """
    hs_dir = traces_dir / "hidden_states"
    if not hs_dir.exists():
        return {"error": "no hidden_states directory"}

    # Collecter les hidden states avec labels
    all_h = []
    labels_tool = []  # 1 si tool call, 0 sinon
    labels_tool_name = []  # nom du tool (ou "none")
    task_ids = []

    for trace in traces:
        for step in trace.get("steps", []):
            hs_path = step.get("hidden_state_path")
            if hs_path and Path(hs_path).exists():
                h = np.load(hs_path)
                all_h.append(h)
                has_tool = step.get("action") is not None
                labels_tool.append(1 if has_tool else 0)
                labels_tool_name.append(step.get("action", "none") or "none")
                task_ids.append(trace.get("task_id", ""))

    if len(all_h) < 10:
        return {"n_states": len(all_h), "error": "insufficient hidden states"}

    H = np.stack(all_h)
    labels_tool = np.array(labels_tool)
    n_total = len(H)
    n_tool = int(labels_tool.sum())

    results = {
        "n_states": n_total,
        "n_tool_calls": n_tool,
        "n_non_tool": n_total - n_tool,
        "hidden_dim": H.shape[1],
    }

    # ── Séparation tool vs non-tool ──
    if n_tool >= 3 and (n_total - n_tool) >= 3:
        from sklearn.metrics import roc_auc_score

        h_tool = H[labels_tool == 1]
        h_notool = H[labels_tool == 0]
        diff = h_tool.mean(axis=0) - h_notool.mean(axis=0)
        diff_norm = diff / (np.linalg.norm(diff) + 1e-12)

        proj = H @ diff_norm
        auc = roc_auc_score(labels_tool, proj)
        auc_inv = roc_auc_score(labels_tool, -proj)
        best_auc = max(auc, auc_inv)

        results["tool_vs_notool_auc"] = float(best_auc)
        results["separation_norm"] = float(np.linalg.norm(diff))

        # Norme moyenne par groupe
        results["mean_norm_tool"] = float(np.mean(np.linalg.norm(h_tool, axis=1)))
        results["mean_norm_notool"] = float(np.mean(np.linalg.norm(h_notool, axis=1)))

    # ── Clustering par tool ──
    tool_names_unique = sorted(set(labels_tool_name) - {"none"})
    if len(tool_names_unique) >= 3:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import LabelEncoder

        # Seulement les steps avec tool call
        mask = np.array([t != "none" for t in labels_tool_name])
        H_tools = H[mask]
        names_tools = [labels_tool_name[i] for i in range(len(labels_tool_name)) if mask[i]]

        if len(set(names_tools)) >= 2 and len(H_tools) >= 5:
            le = LabelEncoder()
            tool_labels = le.fit_transform(names_tools)
            sil = float(silhouette_score(H_tools, tool_labels, metric="cosine"))
            results["tool_cluster_silhouette"] = sil
            results["n_unique_tools_used"] = len(set(names_tools))

    # ── PCA globale ──
    from sklearn.decomposition import PCA
    n_comp = min(10, len(H) - 1, H.shape[1])
    pca = PCA(n_components=n_comp)
    H_pca = pca.fit_transform(H)
    results["pca_explained_variance"] = pca.explained_variance_ratio_[:5].tolist()

    # ── Trajectoire par tâche ──
    trajectories = {}
    current_task = None
    current_traj = []
    for i, tid in enumerate(task_ids):
        if tid != current_task:
            if current_task and len(current_traj) >= 2:
                traj = np.stack(current_traj)
                # Distance parcourue dans l'espace latent
                deltas = np.diff(traj, axis=0)
                total_dist = float(np.sum(np.linalg.norm(deltas, axis=1)))
                # Direction nette
                net_dir = traj[-1] - traj[0]
                net_dist = float(np.linalg.norm(net_dir))
                # Ratio (1.0 = ligne droite, > 1.0 = détour)
                straightness = net_dist / (total_dist + 1e-12)
                trajectories[current_task] = {
                    "n_steps": len(current_traj),
                    "total_distance": total_dist,
                    "net_distance": net_dist,
                    "straightness": straightness,
                }
            current_task = tid
            current_traj = [H_pca[i]]
        else:
            current_traj.append(H_pca[i])

    # Dernier
    if current_task and len(current_traj) >= 2:
        traj = np.stack(current_traj)
        deltas = np.diff(traj, axis=0)
        total_dist = float(np.sum(np.linalg.norm(deltas, axis=1)))
        net_dir = traj[-1] - traj[0]
        net_dist = float(np.linalg.norm(net_dir))
        trajectories[current_task] = {
            "n_steps": len(current_traj),
            "total_distance": total_dist,
            "net_distance": net_dist,
            "straightness": net_dist / (total_dist + 1e-12),
        }

    if trajectories:
        straightness_values = [t["straightness"] for t in trajectories.values()]
        results["mean_trajectory_straightness"] = float(np.mean(straightness_values))
        results["trajectories"] = trajectories

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. Métriques heuristiques améliorées
# ═══════════════════════════════════════════════════════════════════════════

def compute_improved_metrics(traces: list[dict]) -> dict:
    """Métriques complémentaires calculées sans LLM."""
    from baselines.tasks import TASK_MAP

    results = {
        "total_tasks": len(traces),
        "per_task": {},
    }

    # Métriques globales
    total_steps = 0
    total_tool_calls = 0
    total_time_ms = 0
    unique_tools_used = set()
    tool_call_positions = []  # à quel step les tools sont appelés

    for trace in traces:
        tid = trace.get("task_id", "")
        task_info = TASK_MAP.get(tid)

        steps = trace.get("steps", [])
        tools_used = trace.get("tools_used", [])
        expected = trace.get("expected_tools", [])
        n_steps = len(steps)
        n_tools = trace.get("num_tool_calls", 0)

        total_steps += n_steps
        total_tool_calls += n_tools
        total_time_ms += trace.get("total_time_ms", 0)
        unique_tools_used.update(tools_used)

        # Position des tool calls
        for i, step in enumerate(steps):
            if step.get("action"):
                tool_call_positions.append(i / max(1, n_steps - 1) if n_steps > 1 else 0)

        # Per-task metrics
        task_metrics = {
            "n_steps": n_steps,
            "n_tool_calls": n_tools,
            "tools_used": tools_used,
            "expected_tools": expected,
            "has_final_answer": trace.get("final_answer") is not None,
            "time_ms": trace.get("total_time_ms", 0),
        }

        # Tool precision / recall
        if expected:
            used_set = set(tools_used)
            expected_set = set(expected)
            tp = len(used_set & expected_set)
            task_metrics["tool_precision"] = tp / len(used_set) if used_set else 0
            task_metrics["tool_recall"] = tp / len(expected_set) if expected_set else 1.0
        else:
            # Pas de tools attendus — succès si pas de tool call
            task_metrics["tool_precision"] = 1.0 if not tools_used else 0.0
            task_metrics["tool_recall"] = 1.0

        # Détection de boucle fine
        if n_steps >= 3:
            consecutive_same = 0
            max_consecutive = 0
            for i in range(1, len(steps)):
                if (steps[i].get("action") == steps[i - 1].get("action")
                        and steps[i].get("action") is not None):
                    consecutive_same += 1
                    max_consecutive = max(max_consecutive, consecutive_same)
                else:
                    consecutive_same = 0
            task_metrics["max_consecutive_same_tool"] = max_consecutive

        # Erreurs dans les observations
        n_errors = sum(1 for s in steps if s.get("observation") and "ERROR" in str(s.get("observation", "")))
        task_metrics["n_tool_errors"] = n_errors

        results["per_task"][tid] = task_metrics

    # Agrégats
    n = len(traces) or 1
    results["global"] = {
        "mean_steps": total_steps / n,
        "mean_tool_calls": total_tool_calls / n,
        "mean_time_ms": total_time_ms / n,
        "unique_tools_used": sorted(unique_tools_used),
        "n_unique_tools": len(unique_tools_used),
        "tool_call_position_mean": float(np.mean(tool_call_positions)) if tool_call_positions else 0,
        "tool_call_position_std": float(np.std(tool_call_positions)) if tool_call_positions else 0,
    }

    # Tool usage frequency
    all_tools = [t for trace in traces for t in trace.get("tools_used", [])]
    tool_freq = Counter(all_tools)
    results["tool_frequency"] = dict(tool_freq.most_common())

    # Success rate par catégorie
    cat_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for trace in traces:
        cat = trace.get("task_category", "unknown")
        cat_stats[cat]["total"] += 1
        if trace.get("final_answer") is not None and trace.get("error") is None:
            cat_stats[cat]["success"] += 1
    results["success_by_category"] = {
        cat: {**stats, "rate": stats["success"] / max(1, stats["total"])}
        for cat, stats in cat_stats.items()
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. Rapport combiné
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    traces: list[dict],
    judgments: list[dict],
    hs_analysis: dict,
    improved_metrics: dict,
    output_dir: Path,
):
    """Génère le rapport final combiné."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_traces": len(traces),
    }

    # ── Section 1 : LLM Judge ──
    if judgments:
        valid_judgments = [j for j in judgments if j.get("task_completed") is not None]
        if valid_judgments:
            n_completed = sum(1 for j in valid_judgments if j.get("task_completed"))
            judge_failure_modes = Counter(j.get("failure_mode", "unknown") for j in valid_judgments)

            avg_tool_selection = np.mean([j.get("correct_tool_selection", 0) for j in valid_judgments if "correct_tool_selection" in j])
            avg_efficiency = np.mean([j.get("efficiency", 0) for j in valid_judgments if "efficiency" in j])
            avg_answer_quality = np.mean([j.get("answer_quality", 0) for j in valid_judgments if "answer_quality" in j])

            # Tâches où le juge dit qu'un tool n'était pas nécessaire
            no_tool_needed = sum(1 for j in valid_judgments if j.get("tool_usage_needed") == False)

            report["judge"] = {
                "n_judged": len(valid_judgments),
                "n_parse_errors": len(judgments) - len(valid_judgments),
                "completion_rate": n_completed / len(valid_judgments),
                "avg_tool_selection": float(avg_tool_selection),
                "avg_efficiency": float(avg_efficiency),
                "avg_answer_quality": float(avg_answer_quality),
                "no_tool_needed_count": no_tool_needed,
                "failure_modes": dict(judge_failure_modes.most_common()),
                "per_task": {j["task_id"]: j for j in judgments},
            }

    # ── Section 2 : Hidden States ──
    report["hidden_states"] = hs_analysis

    # ── Section 3 : Métriques améliorées ──
    report["metrics"] = improved_metrics

    # ── Comparaison heuristique vs juge ──
    if judgments:
        valid_judgments_map = {j["task_id"]: j for j in judgments if "task_id" in j}
        agreements = 0
        disagreements = []
        for trace in traces:
            tid = trace.get("task_id", "")
            judge = valid_judgments_map.get(tid, {})
            heuristic_success = trace.get("final_answer") is not None and trace.get("error") is None
            judge_success = judge.get("task_completed")

            if judge_success is not None:
                if heuristic_success == judge_success:
                    agreements += 1
                else:
                    disagreements.append({
                        "task_id": tid,
                        "heuristic": heuristic_success,
                        "judge": judge_success,
                        "judge_explanation": judge.get("explanation", ""),
                        "judge_failure_mode": judge.get("failure_mode", ""),
                    })

        total_compared = agreements + len(disagreements)
        report["heuristic_vs_judge"] = {
            "agreement_rate": agreements / max(1, total_compared),
            "n_agreements": agreements,
            "n_disagreements": len(disagreements),
            "disagreements": disagreements,
        }

    # Sauvegarder
    with open(output_dir / "deep_evaluation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Print rapport ──
    print("\n" + "=" * 80)
    print("ÉVALUATION APPROFONDIE — RAPPORT COMBINÉ")
    print("=" * 80)

    if "judge" in report:
        j = report["judge"]
        print(f"\n─── LLM-as-Judge ({j['n_judged']} tâches) ───")
        print(f"  Taux de complétion (juge) : {j['completion_rate']:.1%}")
        print(f"  Qualité réponse (0-3)     : {j['avg_answer_quality']:.2f}")
        print(f"  Sélection tool (0-3)      : {j['avg_tool_selection']:.2f}")
        print(f"  Efficience (0-3)          : {j['avg_efficiency']:.2f}")
        print(f"  Tâches sans besoin tool   : {j['no_tool_needed_count']}")
        print(f"  Modes d'échec (juge)      :")
        for mode, count in sorted(j["failure_modes"].items(), key=lambda x: -x[1]):
            print(f"    {mode:25s} : {count}")

    if "heuristic_vs_judge" in report:
        hvj = report["heuristic_vs_judge"]
        print(f"\n─── Heuristique vs Juge ───")
        print(f"  Accord : {hvj['agreement_rate']:.1%} ({hvj['n_agreements']}/{hvj['n_agreements'] + hvj['n_disagreements']})")
        if hvj["disagreements"]:
            print(f"  Désaccords :")
            for d in hvj["disagreements"][:10]:
                h_str = "✓" if d["heuristic"] else "✗"
                j_str = "✓" if d["judge"] else "✗"
                print(f"    {d['task_id']}: heuristique={h_str} juge={j_str} — {d['judge_explanation'][:80]}")

    hs = report.get("hidden_states", {})
    if "tool_vs_notool_auc" in hs:
        print(f"\n─── Hidden States ───")
        print(f"  AUC tool vs non-tool      : {hs['tool_vs_notool_auc']:.3f}")
        if "tool_cluster_silhouette" in hs:
            print(f"  Silhouette par tool       : {hs['tool_cluster_silhouette']:.3f}")
        if "mean_trajectory_straightness" in hs:
            print(f"  Rectitude trajectoire     : {hs['mean_trajectory_straightness']:.3f}")
        print(f"  PCA variance (top 5)      : {hs.get('pca_explained_variance', [])}")

    m = report.get("metrics", {}).get("global", {})
    if m:
        print(f"\n─── Métriques globales ───")
        print(f"  Steps moyens              : {m.get('mean_steps', 0):.1f}")
        print(f"  Tool calls moyens         : {m.get('mean_tool_calls', 0):.1f}")
        print(f"  Temps moyen (ms)          : {m.get('mean_time_ms', 0):.0f}")
        print(f"  Tools uniques utilisés    : {m.get('n_unique_tools', 0)}")

    print("=" * 80)
    print(f"\nRapport sauvegardé : {output_dir / 'deep_evaluation.json'}")

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Deep evaluation of Sprint 0 traces")
    parser.add_argument("--traces-dir", type=str, required=True,
                        help="Directory containing traces.json and hidden_states/")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3-4B",
                        help="Model to use as LLM judge")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-judge (only heuristic + geometric)")
    parser.add_argument("--skip-hidden-states", action="store_true",
                        help="Skip hidden state analysis")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    traces_dir = Path(args.traces_dir)

    print("=" * 80)
    print("EVALUATE_DEEP — Évaluation post-hoc approfondie")
    print("=" * 80)

    # 1. Charger
    traces, existing = load_traces(traces_dir)

    # 2. Métriques heuristiques améliorées (toujours, gratuit)
    sys.path.insert(0, str(Path(__file__).parent))
    improved_metrics = compute_improved_metrics(traces)

    # 3. Hidden states (si disponibles)
    hs_analysis = {}
    if not args.skip_hidden_states:
        logger.info("Analyzing hidden states...")
        hs_analysis = analyze_hidden_states(traces_dir, traces)

    # 4. LLM Judge (optionnel)
    judgments = []
    if not args.skip_judge:
        logger.info(f"Running LLM judge with {args.judge_model}...")
        judgments = run_judge_batch(traces, args.judge_model, args.device)
    else:
        logger.info("Skipping LLM judge")

    # 5. Rapport
    report = generate_report(traces, judgments, hs_analysis, improved_metrics, traces_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
