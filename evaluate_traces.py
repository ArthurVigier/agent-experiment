"""
evaluate_traces.py
Évaluation post-hoc des traces Sprint 0 avec LLM-as-a-judge.

Réanalyse toutes les traces sauvegardées avec :
  1. Reclassification des succès/échecs par un LLM juge (plus fiable que le pattern matching)
  2. Évaluation de la qualité des réponses finales (pas juste "a-t-il fini")
  3. Analyse des hidden states si disponibles
  4. Rapport détaillé avec métriques enrichies

Usage (sur le pod, après Sprint 0A) :
  # Avec le même modèle que l'agent (self-judge, rapide)
  python evaluate_traces.py --traces-dir results/sprint0_test/qwen3-8b --judge self

  # Avec un modèle juge séparé (plus fiable, charge un 2ème modèle)
  python evaluate_traces.py --traces-dir results/sprint0_test/qwen3-8b --judge Qwen/Qwen3-8B

  # Sans LLM judge (analyse structurelle seulement, pas de GPU)
  python evaluate_traces.py --traces-dir results/sprint0_test/qwen3-8b --judge none

Convention : docstrings français, code anglais.
"""

import argparse
import json
import logging
import sys
import time
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Chargement des données
# ═══════════════════════════════════════════════════════════════════════════

def load_traces(traces_dir: Path) -> tuple[list[dict], dict]:
    """Charge les traces et le résumé."""
    traces_path = traces_dir / "traces.json"
    if not traces_path.exists():
        raise FileNotFoundError(f"Traces not found: {traces_path}")

    with open(traces_path) as f:
        traces = json.load(f)

    summary_path = traces_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    logger.info(f"Loaded {len(traces)} traces from {traces_dir}")
    return traces, summary


def load_tasks() -> dict:
    """Charge les définitions de tâches pour le ground truth."""
    sys.path.insert(0, str(Path(__file__).parent))
    from baselines.tasks import ALL_TASKS
    return {t.id: {
        "id": t.id,
        "prompt": t.prompt,
        "category": t.category,
        "expected_tools": t.expected_tools,
        "complexity": t.complexity,
        "success_hint": t.success_hint,
    } for t in ALL_TASKS}


# ═══════════════════════════════════════════════════════════════════════════
# LLM-as-a-Judge
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for AI agent tool-use traces. 
You will be given a task, the expected behavior, and the agent's actual execution trace.

Evaluate the trace on these dimensions and respond ONLY with a valid JSON object (no markdown, no backticks):

{
  "task_completed": true/false,
  "correct_tool_used": true/false,
  "tool_usage_quality": "optimal" | "acceptable" | "suboptimal" | "wrong" | "none",
  "answer_quality": "correct" | "partially_correct" | "incorrect" | "no_answer",
  "failure_mode": "success" | "wrong_tool" | "no_tool" | "wrong_params" | "loop" | "hallucinated_tool" | "premature_stop" | "wrong_timing" | "repeated_error" | "max_steps",
  "efficiency": "optimal" | "acceptable" | "wasteful",
  "reasoning_quality": "good" | "adequate" | "poor",
  "explanation": "Brief explanation of your evaluation (1-2 sentences)"
}

Rules:
- task_completed: did the agent achieve what was asked?
- correct_tool_used: did the agent use an appropriate tool from the expected list?
- tool_usage_quality: "optimal" = right tool, right params, minimal steps. "none" = no tool used when one was needed.
- If the task can be correctly answered WITHOUT a tool (e.g. simple math), and the agent answered correctly without one, that's "acceptable" tool usage and the task IS completed.
- answer_quality: evaluate the final answer's correctness, not just whether the agent finished.
- efficiency: "optimal" = minimum steps needed. "wasteful" = unnecessary tool calls or retries.
"""


def build_judge_prompt(task: dict, trace: dict) -> str:
    """Construit le prompt pour le juge LLM."""
    # Résumer la trace
    steps_summary = []
    for step in trace.get("steps", []):
        s = f"Step {step['step_idx']}:"
        if step.get("thought"):
            s += f" Thought: {step['thought'][:200]}"
        if step.get("action"):
            params_str = json.dumps(step.get("action_params", {}))[:150]
            s += f" → Action: {step['action']}({params_str})"
        if step.get("observation"):
            s += f" → Observation: {step['observation'][:200]}"
        steps_summary.append(s)

    trace_text = "\n".join(steps_summary) if steps_summary else "No steps recorded."

    final_answer = trace.get("final_answer", "No final answer provided.")

    return f"""TASK: {task['prompt']}

EXPECTED TOOLS: {', '.join(task['expected_tools']) if task['expected_tools'] else 'None required (can answer directly)'}
SUCCESS CRITERIA: {task['success_hint']}

AGENT TRACE ({len(trace.get('steps', []))} steps, {trace.get('num_tool_calls', 0)} tool calls):
{trace_text}

FINAL ANSWER: {final_answer}

TOOLS USED: {', '.join(trace.get('tools_used', [])) if trace.get('tools_used') else 'None'}
ERROR: {trace.get('error', 'None')}

Evaluate this trace. Respond with JSON only."""


class LLMJudge:
    """Évaluateur LLM pour les traces d'agent."""

    def __init__(self, model=None, tokenizer=None, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_model_id(cls, model_id: str, device: str = "cuda"):
        """Charge un modèle juge depuis HuggingFace."""
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
        logger.info("Judge model loaded")
        return cls(model=model, tokenizer=tokenizer, device=device)

    def evaluate(self, task: dict, trace: dict) -> dict:
        """Évalue une trace avec le LLM juge."""
        import torch

        prompt = build_judge_prompt(task, trace)

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"System: {JUDGE_SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = self.tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()

        return self._parse_judge_response(generated)

    def _parse_judge_response(self, response: str) -> dict:
        """Parse la réponse JSON du juge."""
        # Nettoyer la réponse
        response = response.strip()

        # Retirer les backticks markdown si présents
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'^```\s*', '', response)
        response = re.sub(r'\s*```$', '', response)

        # Retirer le thinking de Qwen3 si présent
        if "<think>" in response:
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Essayer d'extraire le JSON
        try:
            # Chercher le premier { ... }
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass

        # Fallback
        logger.warning(f"Could not parse judge response: {response[:200]}")
        return {
            "task_completed": None,
            "correct_tool_used": None,
            "tool_usage_quality": "unknown",
            "answer_quality": "unknown",
            "failure_mode": "unknown",
            "efficiency": "unknown",
            "reasoning_quality": "unknown",
            "explanation": f"Parse error: {response[:200]}",
        }


# ═══════════════════════════════════════════════════════════════════════════
# Analyse structurelle (sans LLM, toujours exécutée)
# ═══════════════════════════════════════════════════════════════════════════

def structural_analysis(traces: list[dict], tasks: dict) -> dict:
    """Analyse structurelle des traces sans LLM."""

    results = {
        "total": len(traces),
        "with_final_answer": sum(1 for t in traces if t.get("final_answer")),
        "with_error": sum(1 for t in traces if t.get("error")),
        "total_tool_calls": sum(t.get("num_tool_calls", 0) for t in traces),
        "total_steps": sum(len(t.get("steps", [])) for t in traces),
    }

    # Steps distribution
    step_counts = [len(t.get("steps", [])) for t in traces]
    tool_counts = [t.get("num_tool_calls", 0) for t in traces]
    results["steps_distribution"] = {
        "mean": float(np.mean(step_counts)),
        "median": float(np.median(step_counts)),
        "std": float(np.std(step_counts)),
        "min": int(np.min(step_counts)),
        "max": int(np.max(step_counts)),
    }
    results["tool_calls_distribution"] = {
        "mean": float(np.mean(tool_counts)),
        "median": float(np.median(tool_counts)),
        "zero_tool_calls": sum(1 for c in tool_counts if c == 0),
    }

    # Timing
    times = [t.get("total_time_ms", 0) for t in traces if t.get("total_time_ms", 0) > 0]
    if times:
        results["timing"] = {
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "total_s": float(np.sum(times) / 1000),
            "fastest_task": min(zip(times, [t.get("task_id", "?") for t in traces]))[1],
            "slowest_task": max(zip(times, [t.get("task_id", "?") for t in traces]))[1],
        }

    # Tool usage patterns
    all_tools_used = Counter()
    for t in traces:
        for tool in t.get("tools_used", []):
            all_tools_used[tool] += 1
    results["tool_frequency"] = dict(all_tools_used.most_common())

    # Tools jamais utilisés
    all_expected = set()
    for t in traces:
        task = tasks.get(t.get("task_id", ""))
        if task:
            all_expected.update(task.get("expected_tools", []))
    results["tools_never_used"] = sorted(all_expected - set(all_tools_used.keys()))

    # Par catégorie
    cat_stats = defaultdict(lambda: {"total": 0, "with_answer": 0, "tool_calls": 0, "steps": 0})
    for t in traces:
        cat = t.get("task_category", "unknown")
        cat_stats[cat]["total"] += 1
        if t.get("final_answer"):
            cat_stats[cat]["with_answer"] += 1
        cat_stats[cat]["tool_calls"] += t.get("num_tool_calls", 0)
        cat_stats[cat]["steps"] += len(t.get("steps", []))
    results["by_category"] = {
        cat: {
            **stats,
            "answer_rate": stats["with_answer"] / stats["total"] if stats["total"] > 0 else 0,
            "avg_tool_calls": stats["tool_calls"] / stats["total"] if stats["total"] > 0 else 0,
            "avg_steps": stats["steps"] / stats["total"] if stats["total"] > 0 else 0,
        }
        for cat, stats in cat_stats.items()
    }

    # Patterns de tool calls (séquences fréquentes)
    sequences = []
    for t in traces:
        tools = t.get("tools_used", [])
        if tools:
            sequences.append(" → ".join(tools))
    results["tool_sequences"] = dict(Counter(sequences).most_common(10))

    # Taux de "no tool when tool was expected"
    no_tool_when_expected = 0
    for t in traces:
        task = tasks.get(t.get("task_id", ""))
        if task and task.get("expected_tools") and t.get("num_tool_calls", 0) == 0:
            no_tool_when_expected += 1
    results["no_tool_when_expected"] = no_tool_when_expected
    results["no_tool_rate"] = no_tool_when_expected / len(traces) if traces else 0

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analyse des hidden states
# ═══════════════════════════════════════════════════════════════════════════

def hidden_state_analysis(traces_dir: Path, traces: list[dict]) -> dict:
    """Analyse les hidden states sauvegardés."""
    hs_dir = traces_dir / "hidden_states"
    if not hs_dir.exists():
        return {"error": "no hidden states directory"}

    hs_files = sorted(hs_dir.glob("*.npy"))
    if not hs_files:
        return {"error": "no hidden state files"}

    results = {"n_files": len(hs_files)}

    # Charger un subset pour l'analyse (pas tout si c'est gros)
    max_files = min(200, len(hs_files))
    hidden_states = []
    labels = []  # 1 si tool call, 0 sinon
    tool_names = []

    for trace in traces:
        for step in trace.get("steps", []):
            hs_path = step.get("hidden_state_path")
            if hs_path and Path(hs_path).exists() and len(hidden_states) < max_files:
                h = np.load(hs_path)
                hidden_states.append(h)
                has_tool = step.get("action") is not None
                labels.append(1 if has_tool else 0)
                tool_names.append(step.get("action", "none"))

    if len(hidden_states) < 10:
        results["error"] = f"insufficient hidden states: {len(hidden_states)}"
        return results

    H = np.stack(hidden_states)
    y = np.array(labels)
    results["n_states"] = len(H)
    results["n_tool_calls"] = int(y.sum())
    results["n_no_tool"] = int((1 - y).sum())
    results["hidden_dim"] = H.shape[1]

    # Norme moyenne des hidden states
    norms = np.linalg.norm(H, axis=1)
    results["hidden_norm_mean"] = float(norms.mean())
    results["hidden_norm_std"] = float(norms.std())

    # Norme tool vs no-tool
    if y.sum() > 0 and (1 - y).sum() > 0:
        results["norm_tool"] = float(norms[y == 1].mean())
        results["norm_no_tool"] = float(norms[y == 0].mean())

    # PCA
    from sklearn.decomposition import PCA
    n_comp = min(10, len(H) - 1, H.shape[1])
    pca = PCA(n_components=n_comp)
    H_pca = pca.fit_transform(H)
    results["pca_variance"] = pca.explained_variance_ratio_[:5].tolist()

    # AUC de chaque PC pour discriminer tool/no-tool
    if len(set(y)) > 1:
        from sklearn.metrics import roc_auc_score
        pc_aucs = []
        for i in range(min(5, H_pca.shape[1])):
            auc = roc_auc_score(y, H_pca[:, i])
            auc_inv = roc_auc_score(y, -H_pca[:, i])
            pc_aucs.append(float(max(auc, auc_inv)))
        results["pc_aucs_tool_vs_notool"] = pc_aucs

        # Direction mean-diff
        h_tool = H[y == 1]
        h_notool = H[y == 0]
        diff = h_tool.mean(axis=0) - h_notool.mean(axis=0)
        diff_norm = diff / (np.linalg.norm(diff) + 1e-12)
        proj = H @ diff_norm
        auc_diff = roc_auc_score(y, proj)
        auc_diff_inv = roc_auc_score(y, -proj)
        results["auc_mean_diff"] = float(max(auc_diff, auc_diff_inv))

    # Clustering par tool
    tool_counts = Counter(tool_names)
    if len([t for t, c in tool_counts.items() if t != "none" and c >= 2]) >= 2:
        from sklearn.metrics import silhouette_score
        # Filtrer les tools avec assez d'occurrences
        valid_mask = np.array([
            tool_counts.get(t, 0) >= 2 and t != "none"
            for t in tool_names
        ])
        if valid_mask.sum() >= 5:
            H_valid = H[valid_mask]
            tools_valid = [t for t, v in zip(tool_names, valid_mask) if v]
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            tool_labels = le.fit_transform(tools_valid)
            if len(set(tool_labels)) > 1:
                sil = silhouette_score(H_valid, tool_labels, metric="cosine")
                results["tool_silhouette_hidden_states"] = float(sil)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Rapport final
# ═══════════════════════════════════════════════════════════════════════════

def print_evaluation_report(
    structural: dict,
    judge_results: list[dict],
    hs_analysis: dict,
    tasks: dict,
    traces: list[dict],
):
    """Affiche le rapport d'évaluation complet."""

    print("\n" + "█" * 70)
    print("█  ÉVALUATION POST-HOC — SPRINT 0 TRACES")
    print("█" * 70)

    # ── Analyse structurelle ──
    print(f"\n{'═' * 70}")
    print("1. ANALYSE STRUCTURELLE")
    print("═" * 70)
    print(f"  Traces totales       : {structural['total']}")
    print(f"  Avec réponse finale  : {structural['with_final_answer']}")
    print(f"  Avec erreur          : {structural['with_error']}")
    print(f"  Tool calls total     : {structural['total_tool_calls']}")
    print(f"  No-tool rate         : {structural['no_tool_rate']:.1%} "
          f"({structural['no_tool_when_expected']} tâches)")

    print(f"\n  Steps : mean={structural['steps_distribution']['mean']:.1f}, "
          f"median={structural['steps_distribution']['median']:.0f}, "
          f"max={structural['steps_distribution']['max']}")

    if "timing" in structural:
        t = structural["timing"]
        print(f"  Temps : mean={t['mean_ms']:.0f}ms, total={t['total_s']:.0f}s")
        print(f"  Plus rapide : {t['fastest_task']}, plus lent : {t['slowest_task']}")

    print(f"\n  Par catégorie :")
    for cat, stats in sorted(structural.get("by_category", {}).items()):
        print(f"    {cat:15s}  answer_rate={stats['answer_rate']:.0%}  "
              f"avg_tools={stats['avg_tool_calls']:.1f}  avg_steps={stats['avg_steps']:.1f}")

    if structural.get("tool_sequences"):
        print(f"\n  Top séquences de tools :")
        for seq, count in list(structural["tool_sequences"].items())[:5]:
            print(f"    {count:3d}× {seq}")

    # ── LLM Judge ──
    if judge_results:
        print(f"\n{'═' * 70}")
        print("2. LLM-AS-A-JUDGE")
        print("═" * 70)

        valid = [j for j in judge_results if j.get("task_completed") is not None]
        if valid:
            completed = sum(1 for j in valid if j["task_completed"])
            correct_tool = sum(1 for j in valid if j.get("correct_tool_used"))

            print(f"\n  Task completed (juge)    : {completed}/{len(valid)} "
                  f"({completed/len(valid):.1%})")
            print(f"  Correct tool used (juge) : {correct_tool}/{len(valid)} "
                  f"({correct_tool/len(valid):.1%})")

            # Distribution tool_usage_quality
            tuq = Counter(j.get("tool_usage_quality", "unknown") for j in valid)
            print(f"\n  Tool usage quality :")
            for quality, count in tuq.most_common():
                bar = "█" * int(count / len(valid) * 30)
                print(f"    {quality:12s}  {count:3d}  ({count/len(valid):5.1%})  {bar}")

            # Distribution answer_quality
            aq = Counter(j.get("answer_quality", "unknown") for j in valid)
            print(f"\n  Answer quality :")
            for quality, count in aq.most_common():
                bar = "█" * int(count / len(valid) * 30)
                print(f"    {quality:18s}  {count:3d}  ({count/len(valid):5.1%})  {bar}")

            # Distribution failure_mode (du juge, pas de l'analyse structurelle)
            fm = Counter(j.get("failure_mode", "unknown") for j in valid)
            print(f"\n  Failure modes (juge) :")
            for mode, count in fm.most_common():
                bar = "█" * int(count / len(valid) * 30)
                print(f"    {mode:18s}  {count:3d}  ({count/len(valid):5.1%})  {bar}")

            # Efficiency
            eff = Counter(j.get("efficiency", "unknown") for j in valid)
            print(f"\n  Efficiency :")
            for e, count in eff.most_common():
                print(f"    {e:12s}  {count:3d}  ({count/len(valid):5.1%})")

            # Comparaison juge vs analyse structurelle
            print(f"\n  Comparaison pattern-matching vs LLM-judge :")
            struct_success = structural["with_final_answer"]
            judge_success = completed
            delta = judge_success - struct_success
            print(f"    Pattern-matching : {struct_success}/{structural['total']} succès")
            print(f"    LLM-judge        : {judge_success}/{len(valid)} succès")
            if delta > 0:
                print(f"    → Le juge est plus permissif ({delta:+d} tâches)")
            elif delta < 0:
                print(f"    → Le juge est plus strict ({delta:+d} tâches)")
            else:
                print(f"    → Accord parfait")

            # Détail des désaccords
            disagreements = []
            for j, trace in zip(judge_results, traces):
                if j.get("task_completed") is None:
                    continue
                struct_ok = trace.get("final_answer") is not None and trace.get("error") is None
                judge_ok = j["task_completed"]
                if struct_ok != judge_ok:
                    disagreements.append({
                        "task_id": trace.get("task_id", "?"),
                        "struct": "success" if struct_ok else "fail",
                        "judge": "success" if judge_ok else "fail",
                        "explanation": j.get("explanation", ""),
                    })
            if disagreements:
                print(f"\n  Désaccords ({len(disagreements)}) :")
                for d in disagreements[:10]:
                    print(f"    {d['task_id']}: struct={d['struct']}, judge={d['judge']}")
                    print(f"      → {d['explanation'][:100]}")

    # ── Hidden states ──
    if hs_analysis and "error" not in hs_analysis:
        print(f"\n{'═' * 70}")
        print("3. ANALYSE HIDDEN STATES")
        print("═" * 70)
        print(f"  States analysés : {hs_analysis['n_states']} "
              f"(dim={hs_analysis['hidden_dim']})")
        print(f"  Tool calls      : {hs_analysis['n_tool_calls']}, "
              f"No-tool : {hs_analysis['n_no_tool']}")

        if "auc_mean_diff" in hs_analysis:
            auc = hs_analysis["auc_mean_diff"]
            print(f"  AUC (mean-diff direction) : {auc:.3f}")
            if auc > 0.75:
                print(f"    ✓ Signal d'agentivité fort")
            elif auc > 0.6:
                print(f"    ~ Signal partiel")
            else:
                print(f"    ✗ Signal faible")

        if "pc_aucs_tool_vs_notool" in hs_analysis:
            print(f"  AUC par PC : {['%.3f' % a for a in hs_analysis['pc_aucs_tool_vs_notool']]}")

        if "tool_silhouette_hidden_states" in hs_analysis:
            sil = hs_analysis["tool_silhouette_hidden_states"]
            print(f"  Silhouette tools (hidden states) : {sil:.3f}")
            if sil > 0.3:
                print(f"    ✓ Les tools clustent dans les hidden states")
            elif sil > 0.1:
                print(f"    ~ Clustering partiel")
            else:
                print(f"    ✗ Pas de clustering")

        if "pca_variance" in hs_analysis:
            print(f"  PCA variance : {['%.3f' % v for v in hs_analysis['pca_variance']]}")

    print(f"\n{'█' * 70}")
    print("█  FIN DE L'ÉVALUATION")
    print("█" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Évaluation post-hoc des traces Sprint 0")
    parser.add_argument("--traces-dir", type=str, required=True,
                        help="Répertoire contenant traces.json et hidden_states/")
    parser.add_argument("--judge", type=str, default="self",
                        help="'none' = pas de LLM judge, 'self' = réutiliser le modèle "
                             "du summary.json, ou un model ID HuggingFace")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-judge-tasks", type=int, default=60,
                        help="Nombre max de tâches à évaluer avec le juge (limiter le coût)")
    parser.add_argument("--output", type=str, default=None,
                        help="Fichier de sortie JSON (default: traces_dir/evaluation.json)")
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)
    output_path = Path(args.output) if args.output else traces_dir / "evaluation.json"

    # Charger
    traces, summary = load_traces(traces_dir)
    tasks = load_tasks()

    # 1. Analyse structurelle (toujours, pas de GPU)
    logger.info("Analyse structurelle...")
    structural = structural_analysis(traces, tasks)

    # 2. Analyse hidden states (pas de GPU, juste numpy)
    logger.info("Analyse hidden states...")
    hs_analysis = hidden_state_analysis(traces_dir, traces)

    # 3. LLM-as-a-Judge (optionnel)
    judge_results = []
    if args.judge != "none":
        # Déterminer le modèle juge
        if args.judge == "self":
            judge_model_id = summary.get("model_id", "Qwen/Qwen3-8B")
        else:
            judge_model_id = args.judge

        logger.info(f"Loading LLM judge: {judge_model_id}")
        judge = LLMJudge.from_model_id(judge_model_id, args.device)

        n_to_judge = min(args.max_judge_tasks, len(traces))
        logger.info(f"Evaluating {n_to_judge} traces with LLM judge...")

        for i, trace in enumerate(traces[:n_to_judge]):
            task_id = trace.get("task_id", "")
            task = tasks.get(task_id, {
                "prompt": trace.get("task_prompt", "Unknown"),
                "expected_tools": [],
                "success_hint": "Unknown",
            })

            logger.info(f"  [{i+1}/{n_to_judge}] {task_id}...")
            try:
                result = judge.evaluate(task, trace)
                result["task_id"] = task_id
                judge_results.append(result)

                status = "✓" if result.get("task_completed") else "✗"
                logger.info(f"    {status} quality={result.get('answer_quality', '?')} "
                            f"tool={result.get('tool_usage_quality', '?')}")
            except Exception as e:
                logger.error(f"    Judge error: {e}")
                judge_results.append({"task_id": task_id, "error": str(e)})

        # Décharger le juge
        del judge
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # 4. Rapport
    print_evaluation_report(structural, judge_results, hs_analysis, tasks, traces)

    # 5. Sauvegarder
    evaluation = {
        "structural": structural,
        "judge_results": judge_results,
        "hidden_state_analysis": hs_analysis,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": args.judge,
    }
    with open(output_path, "w") as f:
        json.dump(evaluation, f, indent=2, default=str)
    logger.info(f"Evaluation saved to {output_path}")


if __name__ == "__main__":
    main()
