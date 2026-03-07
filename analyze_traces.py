#!/usr/bin/env python3
"""
analyze_traces.py
Analyse post-hoc des traces Sprint 0 — standalone, pas besoin de GPU/modèle.

Fonctionne uniquement à partir des fichiers JSON produits par run_sprint0.py.
Permet de re-labeler les succès/échecs, relancer le diagnostic, et produire
des rapports détaillés sans relancer les modèles.

Usage :
  # Analyse complète d'un répertoire de résultats
  python analyze_traces.py results/sprint0_test/qwen3-8b/

  # Re-labeler manuellement puis relancer
  python analyze_traces.py results/sprint0_test/qwen3-8b/ --relabel

  # Comparer plusieurs modèles
  python analyze_traces.py results/sprint0_test/*/

  # Exporter en CSV
  python analyze_traces.py results/sprint0_test/qwen3-8b/ --csv

  # Analyser les hidden states (si .npy présents, pour 0C sans modèle)
  python analyze_traces.py results/sprint0_test/qwen3-8b/ --analyze-hidden-states
"""

import argparse
import json
import sys
import csv
import io
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Chargement des données
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TraceRecord:
    """Trace reconstruite depuis le JSON."""
    task_id: str
    task_prompt: str
    task_category: str
    expected_tools: list[str]
    success: bool
    final_answer: Optional[str]
    total_time_ms: float
    num_tool_calls: int
    tools_used: list[str]
    error: Optional[str]
    steps: list[dict]
    # Ajoutés par l'analyse
    failure_mode: str = ""
    relabeled_success: Optional[bool] = None
    notes: str = ""


def load_traces(traces_path: Path) -> list[TraceRecord]:
    """Charge les traces depuis un fichier JSON."""
    with open(traces_path) as f:
        data = json.load(f)

    traces = []
    for d in data:
        tr = TraceRecord(
            task_id=d.get("task_id", ""),
            task_prompt=d.get("task_prompt", ""),
            task_category=d.get("task_category", ""),
            expected_tools=d.get("expected_tools", []),
            success=d.get("success", False) or d.get("final_answer") is not None,
            final_answer=d.get("final_answer"),
            total_time_ms=d.get("total_time_ms", 0),
            num_tool_calls=d.get("num_tool_calls", 0),
            tools_used=d.get("tools_used", []),
            error=d.get("error"),
            steps=d.get("steps", []),
        )
        traces.append(tr)
    return traces


def load_tasks_ground_truth() -> dict:
    """
    Charge le ground truth des tâches.
    Essaie d'importer depuis baselines.tasks, sinon utilise un fallback.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from baselines.tasks import ALL_TASKS
        return {t.id: {
            "expected_tools": t.expected_tools,
            "category": t.category,
            "complexity": t.complexity,
            "prompt": t.prompt,
        } for t in ALL_TASKS}
    except ImportError:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# Classification des échecs
# ═══════════════════════════════════════════════════════════════════════════

def classify_failure(trace: TraceRecord, ground_truth: dict) -> str:
    """Classifie le mode d'échec d'une trace."""
    gt = ground_truth.get(trace.task_id, {})
    expected = set(gt.get("expected_tools", trace.expected_tools))

    # Succès (avec ou sans relabeling)
    success = trace.relabeled_success if trace.relabeled_success is not None else trace.success
    if success:
        if expected and trace.num_tool_calls == 0 and len(expected) > 0:
            return "no_tool"
        used = set(trace.tools_used)
        if expected and not expected.intersection(used) and len(expected) > 0:
            return "wrong_tool"
        return "success"

    # Max steps
    if trace.error == "max_steps_reached":
        if _is_loop(trace):
            return "loop"
        if _has_repeated_errors(trace):
            return "repeated_error"
        if trace.num_tool_calls == 0:
            return "no_tool"
        return "max_steps"

    # Hallucination
    for step in trace.steps:
        if step.get("action") and step.get("observation", ""):
            if "does not exist" in step["observation"]:
                return "hallucinated_tool"

    if _is_loop(trace):
        return "loop"
    if _has_repeated_errors(trace):
        return "repeated_error"
    if trace.num_tool_calls == 0 and expected:
        return "no_tool"

    used = set(trace.tools_used)
    if expected and not expected.intersection(used):
        return "wrong_tool"

    # Mauvais paramètres
    for step in trace.steps:
        if step.get("action") in expected:
            if step.get("observation") and "ERROR" in step.get("observation", ""):
                return "wrong_params"

    if trace.final_answer is None:
        return "max_steps"

    return "success"


def _is_loop(trace: TraceRecord, threshold: int = 3) -> bool:
    tool_calls = [
        (s.get("action"), str(sorted(s.get("action_params", {}).items()))
         if s.get("action_params") else "")
        for s in trace.steps if s.get("action") is not None
    ]
    if len(tool_calls) < threshold:
        return False
    count = 1
    for i in range(1, len(tool_calls)):
        if tool_calls[i] == tool_calls[i - 1]:
            count += 1
            if count >= threshold:
                return True
        else:
            count = 1
    return False


def _has_repeated_errors(trace: TraceRecord, threshold: int = 2) -> bool:
    error_counts: dict[str, int] = {}
    for step in trace.steps:
        if step.get("action") and step.get("observation") and "ERROR" in step.get("observation", ""):
            error_counts[step["action"]] = error_counts.get(step["action"], 0) + 1
    return any(count >= threshold for count in error_counts.values())


# ═══════════════════════════════════════════════════════════════════════════
# Re-labeling interactif
# ═══════════════════════════════════════════════════════════════════════════

def relabel_traces(traces: list[TraceRecord]) -> list[TraceRecord]:
    """
    Permet de re-labeler manuellement les succès/échecs.
    Affiche chaque trace ambiguë et demande confirmation.
    """
    # Identifier les traces ambiguës :
    # - Marquées success mais pas de tool call alors qu'il en fallait
    # - Marquées success mais expected_tools non utilisés
    # - Marquées échec mais ont une final_answer
    ambiguous = []
    for tr in traces:
        expected = set(tr.expected_tools)
        used = set(tr.tools_used)

        is_ambiguous = False
        reason = ""

        # Success sans tool alors qu'il en fallait
        if tr.success and expected and not expected.intersection(used):
            is_ambiguous = True
            reason = f"Success SANS tool attendu (expected: {expected}, used: {used})"

        # Success sans aucun tool call mais des tools attendus
        if tr.success and expected and tr.num_tool_calls == 0:
            is_ambiguous = True
            reason = f"Success SANS aucun tool call (expected: {expected})"

        # Échec mais a une final_answer
        if not tr.success and tr.final_answer:
            is_ambiguous = True
            reason = f"Échec avec final_answer: {tr.final_answer[:80]}..."

        if is_ambiguous:
            ambiguous.append((tr, reason))

    if not ambiguous:
        print("Aucune trace ambiguë à relabeler.")
        return traces

    print(f"\n{'=' * 70}")
    print(f"RE-LABELING : {len(ambiguous)} traces ambiguës")
    print(f"{'=' * 70}")

    for i, (tr, reason) in enumerate(ambiguous):
        print(f"\n─── [{i+1}/{len(ambiguous)}] {tr.task_id} ───")
        print(f"  Prompt  : {tr.task_prompt[:100]}")
        print(f"  Raison  : {reason}")
        print(f"  Tools   : {tr.tools_used}")
        print(f"  Steps   : {len(tr.steps)}")
        if tr.final_answer:
            print(f"  Answer  : {tr.final_answer[:150]}")
        print(f"  Current : {'SUCCESS' if tr.success else 'FAIL'}")

        while True:
            choice = input("  → [s]uccess / [f]ail / [k]eep / [d]etail ? ").strip().lower()
            if choice == 's':
                tr.relabeled_success = True
                print("  → Relabeled: SUCCESS")
                break
            elif choice == 'f':
                tr.relabeled_success = False
                print("  → Relabeled: FAIL")
                break
            elif choice == 'k':
                print("  → Kept as is")
                break
            elif choice == 'd':
                # Afficher les détails des steps
                for j, step in enumerate(tr.steps):
                    print(f"    Step {j}: action={step.get('action')}, "
                          f"thought={step.get('thought', '')[:80]}")
                    if step.get("observation"):
                        print(f"            obs={step['observation'][:100]}")
            else:
                print("  (s/f/k/d)")

    return traces


# ═══════════════════════════════════════════════════════════════════════════
# Analyse complète
# ═══════════════════════════════════════════════════════════════════════════

def full_analysis(traces: list[TraceRecord], ground_truth: dict) -> dict:
    """Analyse complète des traces avec classification des échecs."""

    # Classifier
    for tr in traces:
        tr.failure_mode = classify_failure(tr, ground_truth)

    mode_counts = Counter(tr.failure_mode for tr in traces)
    total = len(traces)
    success_count = mode_counts.get("success", 0)

    # Par catégorie
    cat_stats = defaultdict(lambda: {"total": 0, "success": 0, "failures": Counter()})
    for tr in traces:
        cat = tr.task_category
        cat_stats[cat]["total"] += 1
        if tr.failure_mode == "success":
            cat_stats[cat]["success"] += 1
        else:
            cat_stats[cat]["failures"][tr.failure_mode] += 1

    # Erreurs répétées inter-tâches
    tool_errors = defaultdict(list)
    for tr in traces:
        errored_tools = set()
        for step in tr.steps:
            if (step.get("action") and step.get("observation")
                    and "ERROR" in step.get("observation", "")
                    and step["action"] not in errored_tools):
                errored_tools.add(step["action"])
                tool_errors[step["action"]].append(tr.task_id)
    cross_task_errors = {
        tool: tasks for tool, tasks in tool_errors.items() if len(tasks) >= 2
    }

    # Tools usage
    all_tools_used = Counter()
    for tr in traces:
        for t in tr.tools_used:
            all_tools_used[t] += 1

    # Timing
    times = [tr.total_time_ms for tr in traces]
    steps = [len(tr.steps) for tr in traces]
    tool_calls = [tr.num_tool_calls for tr in traces]

    # No-tool analysis : quelles tâches le modèle résout sans tool ?
    no_tool_success = [tr for tr in traces if tr.failure_mode == "success" and tr.num_tool_calls == 0]
    no_tool_fail = [tr for tr in traces if tr.failure_mode == "no_tool"]

    return {
        "total": total,
        "success_rate": success_count / total if total > 0 else 0,
        "mode_counts": dict(mode_counts),
        "dominant_failure": max(
            ((m, c) for m, c in mode_counts.items() if m != "success"),
            key=lambda x: x[1], default=("none", 0)
        )[0],
        "by_category": {
            cat: {
                "total": s["total"],
                "success_rate": s["success"] / s["total"] if s["total"] > 0 else 0,
                "failures": dict(s["failures"]),
            }
            for cat, s in cat_stats.items()
        },
        "cross_task_errors": cross_task_errors,
        "tool_usage": dict(all_tools_used.most_common()),
        "timing": {
            "mean_time_ms": float(np.mean(times)) if times else 0,
            "median_time_ms": float(np.median(times)) if times else 0,
            "total_time_s": float(sum(times)) / 1000,
        },
        "steps": {
            "mean": float(np.mean(steps)) if steps else 0,
            "median": float(np.median(steps)) if steps else 0,
            "max": max(steps) if steps else 0,
        },
        "tool_calls": {
            "mean": float(np.mean(tool_calls)) if tool_calls else 0,
            "total": sum(tool_calls),
        },
        "no_tool_success": [tr.task_id for tr in no_tool_success],
        "no_tool_fail": [tr.task_id for tr in no_tool_fail],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Rapport
# ═══════════════════════════════════════════════════════════════════════════

def print_report(traces: list[TraceRecord], analysis: dict, model_name: str = ""):
    """Affiche un rapport complet."""
    print(f"\n{'=' * 70}")
    print(f"ANALYSE POST-HOC{f' — {model_name}' if model_name else ''}")
    print(f"{'=' * 70}")

    print(f"\nTaux de succès : {analysis['success_rate']:.1%} "
          f"({int(analysis['success_rate'] * analysis['total'])}/{analysis['total']})")
    print(f"Mode d'échec dominant : {analysis['dominant_failure']}")

    # Modes d'échec
    print(f"\n─── Modes d'échec ───")
    for mode, count in sorted(analysis["mode_counts"].items(), key=lambda x: -x[1]):
        rate = count / analysis["total"]
        bar = "█" * int(rate * 40)
        print(f"  {mode:20s}  {count:3d}  ({rate:5.1%})  {bar}")

    # Par catégorie
    print(f"\n─── Par catégorie ───")
    for cat, stats in sorted(analysis["by_category"].items()):
        fails = stats["failures"]
        top_fail = max(fails.items(), key=lambda x: x[1]) if fails else ("none", 0)
        fail_str = f"  top: {top_fail[0]}={top_fail[1]}" if top_fail[1] > 0 else ""
        print(f"  {cat:15s}  {stats['success_rate']:5.1%}  ({stats['total']} tâches){fail_str}")

    # No-tool analysis
    print(f"\n─── Tâches résolues SANS tool (le modèle calcule seul) ───")
    for tid in analysis["no_tool_success"]:
        tr = next(t for t in traces if t.task_id == tid)
        print(f"  ✓ {tid}: {tr.task_prompt[:70]}...")

    print(f"\n─── Tâches échouées par ABSENCE de tool call ───")
    for tid in analysis["no_tool_fail"]:
        tr = next(t for t in traces if t.task_id == tid)
        print(f"  ✗ {tid}: {tr.task_prompt[:70]}...")

    # Erreurs répétées
    if analysis["cross_task_errors"]:
        print(f"\n─── Erreurs répétées inter-tâches ───")
        for tool, tasks in analysis["cross_task_errors"].items():
            print(f"  {tool:25s}  {len(tasks)} tâches: {', '.join(tasks[:5])}")

    # Tool usage
    print(f"\n─── Outils les plus utilisés ───")
    for tool, count in list(analysis["tool_usage"].items())[:10]:
        print(f"  {tool:25s}  {count}x")

    # Timing
    t = analysis["timing"]
    print(f"\n─── Timing ───")
    print(f"  Temps total    : {t['total_time_s']:.0f}s")
    print(f"  Temps moyen    : {t['mean_time_ms']/1000:.1f}s / tâche")
    print(f"  Steps moyens   : {analysis['steps']['mean']:.1f}")
    print(f"  Tool calls     : {analysis['tool_calls']['mean']:.1f} / tâche")

    # Recommandation
    dominant = analysis["dominant_failure"]
    print(f"\n─── RECOMMANDATION SPRINT 1 ───")
    if dominant in ("wrong_tool", "hallucinated_tool"):
        print("  → 1A : Espace fonctionnel (routing géométrique)")
    elif dominant in ("loop", "wrong_timing", "no_tool"):
        print("  → 1B : Â detector + simulation single-step")
        print("    Le modèle ne détecte pas quand utiliser un tool.")
    elif dominant in ("premature_stop", "max_steps"):
        print("  → 1C : Trajectory planner")
    elif dominant == "repeated_error":
        print("  → Mémoire géométrique (scar buffer)")
    else:
        print("  → Baseline performant")

    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════
# Analyse des hidden states (si disponibles)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_hidden_states(model_dir: Path, traces: list[TraceRecord]) -> dict:
    """
    Analyse Sprint 0C à partir des .npy sauvegardés.
    Pas besoin du modèle — fonctionne sur les hidden states pré-extraits.
    """
    hs_dir = model_dir / "hidden_states"
    if not hs_dir.exists():
        return {"error": "No hidden_states/ directory found"}

    # Collecter les hidden states et labels
    all_h = []
    labels = []  # 1 = tool call, 0 = no tool call

    for tr in traces:
        for step in tr.steps:
            hs_path = step.get("hidden_state_path")
            if hs_path and Path(hs_path).exists():
                h = np.load(hs_path)
                has_tool = step.get("action") is not None
                all_h.append(h)
                labels.append(1 if has_tool else 0)

    if len(all_h) < 10:
        return {"error": f"Too few hidden states: {len(all_h)}"}

    H = np.stack(all_h)
    y = np.array(labels)
    n_tool = int(y.sum())
    n_notool = len(y) - n_tool

    print(f"\n─── Analyse Hidden States ───")
    print(f"  Steps collectés : {len(H)} ({n_tool} tool calls, {n_notool} non-tool)")
    print(f"  Dimension       : {H.shape[1]}")

    results = {
        "n_steps": len(H),
        "n_tool_calls": n_tool,
        "hidden_dim": H.shape[1],
    }

    if n_tool < 2 or n_notool < 2:
        print("  ⚠ Pas assez de diversité pour l'analyse")
        return results

    from sklearn.metrics import roc_auc_score
    from sklearn.decomposition import PCA

    # Direction mean-diff
    h_tool = H[y == 1]
    h_notool = H[y == 0]
    diff = h_tool.mean(axis=0) - h_notool.mean(axis=0)
    diff_norm = diff / (np.linalg.norm(diff) + 1e-12)

    proj = H @ diff_norm
    auc = roc_auc_score(y, proj)
    auc_inv = roc_auc_score(y, -proj)
    best_auc = max(auc, auc_inv)

    print(f"  AUC (mean-diff) : {best_auc:.3f}")

    # PCA
    pca = PCA(n_components=min(10, len(H) - 1, H.shape[1]))
    H_pca = pca.fit_transform(H)

    pc_aucs = []
    for i in range(min(5, H_pca.shape[1])):
        pc_auc = max(
            roc_auc_score(y, H_pca[:, i]),
            roc_auc_score(y, -H_pca[:, i])
        )
        pc_aucs.append(pc_auc)

    print(f"  AUC par PC      : {['%.3f' % a for a in pc_aucs]}")
    print(f"  PCA var expliquée: {['%.3f' % v for v in pca.explained_variance_ratio_[:5]]}")

    results.update({
        "auc_mean_diff": float(best_auc),
        "pc_aucs": [float(a) for a in pc_aucs],
        "pca_variance": pca.explained_variance_ratio_[:5].tolist(),
    })

    # R̂ si disponible
    r_hat_candidates = list(model_dir.glob("*r_hat*.npy")) + list(model_dir.parent.glob("*r_hat*.npy"))
    for r_hat_path in r_hat_candidates:
        r_hat = np.load(r_hat_path)
        if r_hat.shape[0] == H.shape[1]:
            r_hat = r_hat / (np.linalg.norm(r_hat) + 1e-12)
            proj_r = H @ r_hat
            auc_r = max(roc_auc_score(y, proj_r), roc_auc_score(y, -proj_r))
            cos_a_r = float(np.dot(diff_norm, r_hat))
            print(f"  AUC (R̂)         : {auc_r:.3f}")
            print(f"  cos(Â, R̂)       : {cos_a_r:.3f}")
            results["auc_r_hat"] = float(auc_r)
            results["cos_a_hat_r_hat"] = cos_a_r
            break

    # Verdict
    if best_auc > 0.75:
        print(f"  ✓ Signal d'agentivité FORT")
    elif best_auc > 0.60:
        print(f"  ~ Signal PARTIEL")
    else:
        print(f"  ✗ Signal FAIBLE — cascade de fallback recommandée")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Export CSV
# ═══════════════════════════════════════════════════════════════════════════

def export_csv(traces: list[TraceRecord], output_path: Path):
    """Exporte les traces en CSV pour analyse dans un tableur."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "category", "prompt", "success", "relabeled",
            "failure_mode", "num_steps", "num_tool_calls", "tools_used",
            "expected_tools", "time_ms", "final_answer", "error"
        ])
        for tr in traces:
            writer.writerow([
                tr.task_id,
                tr.task_category,
                tr.task_prompt[:100],
                tr.relabeled_success if tr.relabeled_success is not None else tr.success,
                "relabeled" if tr.relabeled_success is not None else "original",
                tr.failure_mode,
                len(tr.steps),
                tr.num_tool_calls,
                "|".join(tr.tools_used),
                "|".join(tr.expected_tools),
                f"{tr.total_time_ms:.0f}",
                (tr.final_answer or "")[:100],
                tr.error or "",
            ])
    print(f"CSV exporté : {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Comparaison multi-modèle
# ═══════════════════════════════════════════════════════════════════════════

def compare_models(model_dirs: list[Path]):
    """Compare les résultats de plusieurs modèles."""
    gt = load_tasks_ground_truth()
    all_analyses = {}

    for model_dir in model_dirs:
        traces_path = model_dir / "traces.json"
        if not traces_path.exists():
            print(f"⚠ Pas de traces.json dans {model_dir}")
            continue

        model_name = model_dir.name
        traces = load_traces(traces_path)
        analysis = full_analysis(traces, gt)
        all_analyses[model_name] = analysis

    if len(all_analyses) < 2:
        print("Pas assez de modèles pour comparer")
        return

    # Table comparative
    print(f"\n{'=' * 90}")
    print(f"COMPARAISON MULTI-MODÈLE")
    print(f"{'=' * 90}")

    header = f"{'Modèle':<30s} {'Success':>8s} {'no_tool':>8s} {'wrong_t':>8s} {'loop':>6s} {'Steps':>6s} {'Time/task':>10s}"
    print(header)
    print("-" * len(header))

    for name, a in sorted(all_analyses.items()):
        mc = a["mode_counts"]
        tot = a["total"]
        print(f"{name:<30s} "
              f"{a['success_rate']:>7.1%} "
              f"{mc.get('no_tool', 0)/tot:>7.1%} "
              f"{mc.get('wrong_tool', 0)/tot:>7.1%} "
              f"{mc.get('loop', 0)/tot:>5.1%} "
              f"{a['steps']['mean']:>6.1f} "
              f"{a['timing']['mean_time_ms']/1000:>9.1f}s")

    print(f"{'=' * 90}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyse post-hoc des traces Sprint 0")
    parser.add_argument("model_dirs", nargs="+", type=Path,
                        help="Répertoire(s) contenant les traces.json")
    parser.add_argument("--relabel", action="store_true",
                        help="Mode re-labeling interactif")
    parser.add_argument("--csv", action="store_true",
                        help="Exporter en CSV")
    parser.add_argument("--analyze-hidden-states", action="store_true",
                        help="Lancer l'analyse des hidden states (0C)")
    parser.add_argument("--save", action="store_true",
                        help="Sauvegarder l'analyse dans analysis.json")
    args = parser.parse_args()

    gt = load_tasks_ground_truth()

    # Multi-modèle ?
    if len(args.model_dirs) > 1:
        compare_models(args.model_dirs)

    # Analyse par modèle
    for model_dir in args.model_dirs:
        # Trouver traces.json
        if model_dir.is_file() and model_dir.name == "traces.json":
            traces_path = model_dir
            model_dir = model_dir.parent
        elif (model_dir / "traces.json").exists():
            traces_path = model_dir / "traces.json"
        else:
            # Chercher dans les sous-répertoires
            candidates = list(model_dir.glob("*/traces.json"))
            if candidates:
                for c in candidates:
                    print(f"\nFound: {c}")
                    traces = load_traces(c)
                    analysis = full_analysis(traces, gt)
                    print_report(traces, analysis, c.parent.name)
                continue
            else:
                print(f"⚠ Pas de traces.json trouvé dans {model_dir}")
                continue

        traces = load_traces(traces_path)
        model_name = model_dir.name

        # Relabeling ?
        if args.relabel:
            traces = relabel_traces(traces)

        # Analyse
        analysis = full_analysis(traces, gt)
        print_report(traces, analysis, model_name)

        # Hidden states ?
        if args.analyze_hidden_states:
            hs_results = analyze_hidden_states(model_dir, traces)
            analysis["hidden_states"] = hs_results

        # Export CSV ?
        if args.csv:
            csv_path = model_dir / "traces_analysis.csv"
            export_csv(traces, csv_path)

        # Sauvegarder ?
        if args.save:
            out_path = model_dir / "post_hoc_analysis.json"
            with open(out_path, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"Analyse sauvegardée : {out_path}")

            # Sauvegarder aussi les relabels si modifiés
            relabels = {tr.task_id: tr.relabeled_success
                        for tr in traces if tr.relabeled_success is not None}
            if relabels:
                relabel_path = model_dir / "relabels.json"
                with open(relabel_path, "w") as f:
                    json.dump(relabels, f, indent=2)
                print(f"Relabels sauvegardés : {relabel_path}")


if __name__ == "__main__":
    main()
