"""
failure_analysis.py
Diagnostic des modes d'échec — Sprint 0B.

Classifie chaque trace en mode d'échec selon la taxonomie :
  - wrong_tool : le bon tool existait mais l'agent en a choisi un autre
  - wrong_timing : tool bon mais appelé au mauvais moment de la séquence
  - wrong_params : bon tool, bon moment, mauvais arguments
  - no_tool : l'agent n'a pas détecté qu'il devait agir
  - loop : l'agent répète le même tool call sans progresser
  - hallucinated_tool : appel d'un tool qui n'existe pas
  - premature_stop : l'agent s'arrête avant d'avoir fini
  - max_steps : l'agent a épuisé ses steps sans finir
  - success : pas d'échec

Convention : docstrings français, code anglais.
"""

from collections import Counter
from typing import Optional
from baselines.react_agent import Trace, StepRecord
from baselines.tasks import Task


def classify_failure(trace: Trace, task: Task) -> str:
    """
    Classifie le mode d'échec d'une trace.
    
    La classification est hiérarchique : on teste les modes les plus
    graves/spécifiques d'abord, puis les plus généraux.
    """
    # Succès
    if trace.final_answer is not None and trace.error is None:
        # Vérifier que les tools attendus ont été utilisés (si spécifiés)
        if task.expected_tools:
            used = set(trace.tools_used)
            expected = set(task.expected_tools)
            if not expected.intersection(used) and len(expected) > 0:
                # A répondu mais sans utiliser les bons tools
                if trace.num_tool_calls == 0:
                    return "no_tool"
                else:
                    return "wrong_tool"
        return "success"

    # Max steps atteint
    if trace.error == "max_steps_reached":
        # Sous-classifier : pourquoi a-t-on atteint max steps ?
        if _is_loop(trace):
            return "loop"
        if _has_repeated_errors(trace):
            return "repeated_error"
        if trace.num_tool_calls == 0:
            return "no_tool"
        return "max_steps"

    # Hallucination de tool
    for step in trace.steps:
        if step.action and step.observation and "does not exist" in step.observation:
            return "hallucinated_tool"

    # Boucle
    if _is_loop(trace):
        return "loop"

    # Erreur répétée (même tool, même type d'erreur, plus de 2 fois)
    if _has_repeated_errors(trace):
        return "repeated_error"

    # Pas de tool call alors qu'il en fallait
    if trace.num_tool_calls == 0 and task.expected_tools:
        return "no_tool"

    # Mauvais tool
    if task.expected_tools:
        used = set(trace.tools_used)
        expected = set(task.expected_tools)
        if not expected.intersection(used):
            return "wrong_tool"

    # Mauvais timing (les bons tools sont utilisés mais dans le mauvais ordre)
    if task.complexity == "chain" and task.expected_tools:
        if set(trace.tools_used) >= set(task.expected_tools):
            tool_calls = [s.action for s in trace.steps if s.action]
            if tool_calls and tool_calls[0] != task.expected_tools[0]:
                return "wrong_timing"

    # Arrêt prématuré
    if trace.final_answer is not None and task.expected_tools:
        used = set(trace.tools_used)
        expected = set(task.expected_tools)
        if not expected.issubset(used):
            return "premature_stop"

    # Mauvais paramètres
    for step in trace.steps:
        if step.action in (task.expected_tools or []):
            if step.observation and "ERROR" in step.observation:
                return "wrong_params"

    # Fallback
    if trace.final_answer is None:
        return "max_steps"
    
    return "success"


def _is_loop(trace: Trace, threshold: int = 3) -> bool:
    """
    Détecte si l'agent est en boucle.
    
    Critère : le même tool avec les mêmes paramètres est appelé
    plus de `threshold` fois consécutivement.
    """
    if len(trace.steps) < threshold:
        return False

    tool_calls = [
        (s.action, str(sorted(s.action_params.items())) if s.action_params else "")
        for s in trace.steps
        if s.action is not None
    ]

    if len(tool_calls) < threshold:
        return False

    # Vérifier les répétitions consécutives
    count = 1
    for i in range(1, len(tool_calls)):
        if tool_calls[i] == tool_calls[i - 1]:
            count += 1
            if count >= threshold:
                return True
        else:
            count = 1

    return False


def _has_repeated_errors(trace: Trace, threshold: int = 2) -> bool:
    """
    Détecte si l'agent répète le même type d'erreur.
    
    Critère : le même tool produit une erreur (observation contient "ERROR")
    plus de `threshold` fois dans la trace, même avec des paramètres différents.
    
    C'est différent de _is_loop : une boucle répète le même appel exact,
    une erreur répétée fait le même type d'erreur avec des variations.
    """
    if len(trace.steps) < threshold:
        return False

    # Compter les erreurs par tool
    error_counts: dict[str, int] = {}
    for step in trace.steps:
        if step.action and step.observation and "ERROR" in step.observation:
            error_counts[step.action] = error_counts.get(step.action, 0) + 1

    return any(count >= threshold for count in error_counts.values())


def count_repeated_errors_across_traces(traces: list[Trace]) -> dict:
    """
    Compte les erreurs répétées ENTRE les traces (inter-tâche).
    
    Si le même tool produit le même type d'erreur dans plusieurs tâches,
    c'est un signal fort que la mémoire géométrique pourrait aider.
    
    Retourne un dict {tool_name: {"error_count": N, "task_ids": [...]}}
    """
    tool_errors: dict[str, list[str]] = {}  # tool → [task_ids with errors]
    
    for trace in traces:
        tools_errored = set()
        for step in trace.steps:
            if (step.action and step.observation 
                    and "ERROR" in step.observation 
                    and step.action not in tools_errored):
                tools_errored.add(step.action)
                if step.action not in tool_errors:
                    tool_errors[step.action] = []
                tool_errors[step.action].append(trace.task_id)
    
    return {
        tool: {"error_count": len(tasks), "task_ids": tasks}
        for tool, tasks in tool_errors.items()
        if len(tasks) >= 2  # Au moins 2 tâches avec la même erreur
    }


def analyze_traces(traces: list[Trace], tasks: list[Task]) -> dict:
    """
    Analyse complète des traces.
    
    Retourne un dictionnaire avec :
    - Taux de succès global et par catégorie
    - Distribution des modes d'échec
    - Statistiques détaillées
    """
    task_map = {t.id: t for t in tasks}
    
    classifications = {}
    for trace in traces:
        task = task_map.get(trace.task_id)
        if task is None:
            continue
        mode = classify_failure(trace, task)
        classifications[trace.task_id] = mode
        # Annoter aussi chaque step de la trace
        if mode != "success":
            for step in trace.steps:
                if step.failure_mode is None:
                    step.failure_mode = mode

    # Statistiques globales
    mode_counts = Counter(classifications.values())
    total = len(classifications)
    success_count = mode_counts.get("success", 0)

    # Par catégorie
    category_stats = {}
    for task in tasks:
        if task.id not in classifications:
            continue
        cat = task.category
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "success": 0, "failures": Counter()}
        category_stats[cat]["total"] += 1
        mode = classifications[task.id]
        if mode == "success":
            category_stats[cat]["success"] += 1
        else:
            category_stats[cat]["failures"][mode] += 1

    # Par complexité
    complexity_stats = {}
    for task in tasks:
        if task.id not in classifications:
            continue
        comp = task.complexity
        if comp not in complexity_stats:
            complexity_stats[comp] = {"total": 0, "success": 0}
        complexity_stats[comp]["total"] += 1
        if classifications[task.id] == "success":
            complexity_stats[comp]["success"] += 1

    # Tool usage stats
    all_tools_used = Counter()
    all_tools_expected = Counter()
    for trace in traces:
        task = task_map.get(trace.task_id)
        if task:
            for t in trace.tools_used:
                all_tools_used[t] += 1
            for t in task.expected_tools:
                all_tools_expected[t] += 1

    # Steps stats
    step_counts = [len(t.steps) for t in traces]
    tool_call_counts = [t.num_tool_calls for t in traces]

    result = {
        "total_tasks": total,
        "success_rate": success_count / total if total > 0 else 0,
        "failure_modes": {
            mode: {
                "count": count,
                "rate": count / total if total > 0 else 0,
            }
            for mode, count in mode_counts.items()
        },
        "dominant_failure": max(
            ((m, c) for m, c in mode_counts.items() if m != "success"),
            key=lambda x: x[1],
            default=("none", 0),
        )[0],
        "by_category": {
            cat: {
                "total": stats["total"],
                "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
                "top_failure": stats["failures"].most_common(1)[0] if stats["failures"] else ("none", 0),
            }
            for cat, stats in category_stats.items()
        },
        "by_complexity": {
            comp: {
                "total": stats["total"],
                "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
            }
            for comp, stats in complexity_stats.items()
        },
        "tool_usage": {
            "most_used": all_tools_used.most_common(5),
            "most_expected": all_tools_expected.most_common(5),
            "never_used": [
                t for t in all_tools_expected
                if t not in all_tools_used
            ],
        },
        "step_stats": {
            "mean_steps": sum(step_counts) / len(step_counts) if step_counts else 0,
            "mean_tool_calls": sum(tool_call_counts) / len(tool_call_counts) if tool_call_counts else 0,
            "max_steps": max(step_counts) if step_counts else 0,
        },
        "classifications": classifications,
    }

    return result


def print_report(analysis: dict, traces: list = None):
    """Affiche un rapport lisible du diagnostic."""
    print("\n" + "=" * 70)
    print("SPRINT 0B — DIAGNOSTIC D'ÉCHECS")
    print("=" * 70)

    print(f"\nTaux de succès global : {analysis['success_rate']:.1%} "
          f"({int(analysis['success_rate'] * analysis['total_tasks'])}/{analysis['total_tasks']})")

    print(f"\nMode d'échec dominant : {analysis['dominant_failure']}")

    print("\n─── Modes d'échec ───")
    for mode, info in sorted(analysis["failure_modes"].items(), key=lambda x: -x[1]["count"]):
        bar = "█" * int(info["rate"] * 40)
        print(f"  {mode:20s}  {info['count']:3d}  ({info['rate']:5.1%})  {bar}")

    print("\n─── Par catégorie ───")
    for cat, stats in analysis["by_category"].items():
        top_fail = stats["top_failure"]
        fail_str = f"  (top fail: {top_fail[0]}={top_fail[1]})" if top_fail != ("none", 0) else ""
        print(f"  {cat:15s}  {stats['success_rate']:5.1%}  ({stats['total']} tâches){fail_str}")

    print("\n─── Par complexité ───")
    for comp, stats in analysis["by_complexity"].items():
        print(f"  {comp:15s}  {stats['success_rate']:5.1%}  ({stats['total']} tâches)")

    print("\n─── Steps ───")
    ss = analysis["step_stats"]
    print(f"  Steps moyens/tâche : {ss['mean_steps']:.1f}")
    print(f"  Tool calls moyens  : {ss['mean_tool_calls']:.1f}")
    print(f"  Steps max          : {ss['max_steps']}")

    # Recommandation Sprint 1
    dominant = analysis["dominant_failure"]
    print("\n─── RECOMMANDATION SPRINT 1 ───")
    if dominant in ("wrong_tool", "hallucinated_tool"):
        print("  → Variante 1A : Espace fonctionnel (routing géométrique)")
        print("    Le problème principal est la sélection de tool.")
    elif dominant in ("loop", "wrong_timing", "no_tool"):
        print("  → Variante 1B : Â detector + simulation single-step")
        print("    Le problème principal est le timing / détection d'action.")
    elif dominant in ("premature_stop", "max_steps"):
        print("  → Variante 1C : Trajectory planner léger")
        print("    Le problème principal est la planification multi-step.")
    elif dominant == "wrong_params":
        print("  → Hors scope JEPA — focus sur le prompt engineering ou fine-tuning")
    elif dominant == "repeated_error":
        print("  → Mémoire géométrique prioritaire (Sprint 2 scar buffer)")
        print("    L'agent répète les mêmes erreurs — le scar buffer est le fix direct.")
    else:
        print("  → Baseline déjà performant — skip au Sprint 2")

    # Analyse des erreurs répétées inter-tâches
    cross_task_errors = count_repeated_errors_across_traces(traces) if traces else {}
    if cross_task_errors:
        print("\n─── ERREURS RÉPÉTÉES INTER-TÂCHES (potentiel mémoire géométrique) ───")
        for tool, info in sorted(cross_task_errors.items(), key=lambda x: -x[1]["error_count"]):
            print(f"  {tool:25s}  {info['error_count']} tâches avec erreur  "
                  f"(tasks: {', '.join(info['task_ids'][:5])}{'...' if len(info['task_ids']) > 5 else ''})")
        total_cross = sum(info["error_count"] for info in cross_task_errors.values())
        print(f"  Total : {total_cross} erreurs répétées inter-tâches")
        print(f"  → Ces erreurs sont les candidates directes pour le scar buffer")
    
    print("=" * 70)
