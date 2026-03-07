# JEPA-Agent — Geometric Agency for LLM Tool Use

*An AI agent based on Joint Embedding Predictive Architecture.*
Sprint 0: baseline + diagnostics + Â validation + **geometric scaling analysis**.
**Sprint 1: Â detector + gating — 76% success rate ( +26% over baseline)**

> **Note:** All code comments and docstrings in the project are in **French** (a deliberate choice for maintainers), but this README and user-facing interfaces are in English.

---

## 🎯 **Key Results — Sprint 1**

We successfully transformed a geometric signal (`Â`) into an operational **agentivity detector**, achieving:

| Metric | Baseline | Sprint 1 (Â gating) | Δ |
|--------|----------|---------------------|-----|
| **Success rate** | 50% | **76.3%** | **+26.3%** |
| **No-tool rate** | 43% | **2.6%** | **-40.4%** |
| **Wrong tool** | 7% | **2.6%** | **-4.4%** |

📊 **Detailed results available in** [`sprint_1_results/`](sprint_1_results/)

### 🔥 What the LLM judge says:
- **76.3%** tasks completed successfully
- **78.9%** correct tool usage
- **52.6%** tool usage rated "optimal"
- Only **2.6%** "no_tool" failures (down from 43%!)

[See full evaluation →](sprint_1_results/qwen3-8b/evaluation.json)

---

## 🧠 **What We Proved**

| Hypothesis | Validation |
|------------|------------|
| The signal `Â` (agentivity) exists | ✅ AUC > 0.94 across all model sizes |
| It can be extracted geometrically | ✅ Linear probe works with near-perfect separation |
| It can guide an agent in real-time | ✅ **76% success rate** on 40 diverse tasks |
| It solves the "no_tool" problem | ✅ **43% → 2.6%** reduction |

**Key insight:** The hidden state right before a tool call is **linearly separable** from non-tool states — even in small models (1.7B). This geometric property is universal and exploitable.

---

## 🚀 Quick Start (RunPod A100 80GB)

```bash
# Setup
pip install -r requirements.txt
huggingface-cli login  # HF token for gated models (optional for Qwen)

# ─── Sprint 0: Baseline + diagnosis ───
python run_sprint0.py --model Qwen/Qwen3-8B

# ─── Sprint 1: Â detector + gating ───
python run_sprint1.py --sprint0-dir results/sprint0 --model Qwen/Qwen3-8B

# ─── Multi-model scaling analysis ───
python run_sprint0.py --scaling-preset
python run_sprint1.py --sprint0-dir results/sprint0 --scaling-preset
```

---

## 📊 VRAM per model (bf16)

| Model | VRAM | Note |
|-------|------|------|
| Qwen3-0.6B | ~1 GB | Baseline, interesting for max delta |
| Qwen3-1.7B | ~3 GB | Good signal/size trade-off |
| Qwen3-4B | ~6 GB | Intermediate point |
| Qwen3-8B | ~14 GB | Main reference |
| Qwen3-14B | ~28 GB | Optional, if budget allows |

Models are loaded and unloaded **sequentially**—only one in VRAM at a time.

---

## 📁 Output Structure

```
results/
├── sprint0/                          # Baseline & diagnostics
│   ├── scaling_analysis.json         # Geometric scaling across models
│   └── qwen3-8b/
│       ├── traces.json                # Raw traces
│       ├── a_hat_extracted.npy        # Â direction
│       └── hidden_states/             # .npy per step
│
├── sprint1/                          # Â detector results
│   └── qwen3-8b/
│       ├── traces_sprint1.json        # Agent traces with Â gating
│       ├── evaluation.json             # LLM-as-a-judge results
│       └── checkpoint.json             # Resume point
│
└── sprint_1_results/                  # Full analysis & benchmarks
    ├── README.md
    ├── qwen3-8b/
    │   ├── evaluation.json
    │   └── failure_analysis.json
    └── comparison_baseline_vs_sprint1.md
```

---

## 🧠 Project Structure

```
jepa-agent/
├── baselines/
│   ├── tools.py              # 25 tools, 5 categories
│   ├── executors.py          # Real execution (15 real + 5 API + 5 mock)
│   ├── tasks.py              # 60 tasks (40 single, 12 chain, 8 adversarial)
│   ├── react_agent.py        # ReAct agent + hidden state logger
│   └── failure_analysis.py   # Failure taxonomy (Sprint 0B)
├── geometry/                  # Sprint 0C + Sprint 1
│   ├── signal_extraction.py   # Multi-layer Â extraction
│   └── signal_hunt.py         # Fallback cascade for weak signals
├── scripts/
│   ├── run_sprint0.py         # Multi-model baseline runner
│   └── run_sprint1.py         # Â detector + gating (with checkpoint resume)
├── evaluate_traces.py         # LLM-as-a-judge evaluation
├── evaluate_deep.py           # Deep evaluation with multiple judges
└── requirements.txt
```

---

## 🤖 LLM-as-a-Judge Evaluation

We provide two evaluation scripts for nuanced, human-like judgment:

### `evaluate_traces.py`
```bash
python evaluate_traces.py \
    --traces-dir results/sprint1/qwen3-8b \
    --output results/sprint1/qwen3-8b/evaluation.json
```

Evaluates:
- ✅ Task completion
- ✅ Correct tool usage
- ✅ Answer quality
- ✅ Efficiency
- ✅ Reasoning quality

### `evaluate_deep.py`
```bash
python evaluate_deep.py \
    --traces results/sprint1/qwen3-8b/traces_sprint1.json \
    --tasks baselines/tasks.py \
    --output-dir results/sprint1/qwen3-8b/deep_eval/
```

**Why LLM-as-a-Judge?**
Rule-based evaluation misses subtle failures (hallucinations, wrong timing). LLM judges provide **human-level nuance** and cross-validation.

---

## 📈 **Sprint 1 Results — Detailed**

From LLM judge evaluation on 40 single tasks:

```
✅ Success rate: 76.3%
🎯 Correct tool usage: 78.9%
🔧 Tool usage quality:
   optimal     52.6%
   suboptimal  15.8%
   acceptable  10.5%
   wrong       13.2%
   none         7.9%

❌ Failure modes:
   success          71.1%
   max_steps        10.5%
   wrong_params     10.5%
   wrong_tool        2.6%
   no_tool           2.6%
   hallucinated      2.6%
```

**Key achievement:** The "no_tool" failure mode — dominant in baseline (43%) — is now **almost eliminated** (2.6%).

[Full results →](sprint_1_results/qwen3-8b/evaluation.json)

---

## 📝 Requirements

See `requirements.txt` for full list. Main dependencies:
- `torch>=2.2.0`
- `transformers>=4.40.0`
- `accelerate>=0.28.0`
- `duckduckgo-search>=6.0.0` (for real web search)
- `scikit-learn>=1.3.0` (for geometric analysis)

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{jepa-agent2026,
  author = {Arthur Vigier},
  title = {JEPA-Agent: Geometric Agency for LLM Tool Use},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ArthurVigier/jepa-agent}
}
```

---

## 📜 License

Apache 2.0 — free for academic and commercial use, with attribution.
