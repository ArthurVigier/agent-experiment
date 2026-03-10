# Agent-Experiment — Geometric Agency for LLM Tool Use

*An AI agent based on Joint Embedding Predictive Architecture.*
Sprint 0: baseline + diagnostics + Â validation + **geometric scaling analysis**.
**Sprint 1: Â detector + gating — up to 76% success rate (+26 to +49 points over baseline)**

> **Note:** All code comments and docstrings in the project are in **French** (a deliberate choice for maintainers), but this README and user-facing interfaces are in English.

---

## 🎯 **Key Results — Sprint 1 (Complete Scaling Analysis)**

We successfully transformed a geometric signal (`Â`) into an operational **agentivity detector**, with consistent gains across all model sizes:

| Model | Baseline | Sprint 1 (Â gating) | Gain | No-tool before | No-tool after |
|-------|----------|---------------------|------|----------------|---------------|
| **Qwen3-1.7B** | 26% | **75%** | **+49 pts** 🚀 | 73% | **3.1%** |
| **Qwen3-4B** | 26% | **68%** | **+42 pts** 📈 | 73% | **3.2%** |
| **Qwen3-8B** | 50% | **76%** | **+26 pts** 🎯 | 43% | **2.6%** |

📊 **Detailed results available in** [`sprint_1_results/`](sprint_1_results/)

### 🔥 Key insights:
- **The smaller the model, the larger the gain** — geometry compensates for weaker text decoding
- **The "no-tool" problem is solved** across all scales (from 73% → ~3%)
- **Â signal is universal** (AUC > 0.94 on 1.7B, 4B, and 8B)
- **Performance plateaus at ~75-76%** — next challenge is parameter quality

[See full evaluation →](sprint_1_results/qwen3-8b/evaluation.json)

---

## 🧠 **What We Proved**

| Hypothesis | Validation |
|------------|------------|
| The signal `Â` (agentivity) exists | ✅ AUC > 0.94 across all model sizes |
| It can be extracted geometrically | ✅ Linear probe works with near-perfect separation |
| It guides agents in real-time | ✅ **68-76% success rate** on 40 diverse tasks |
| It solves the "no-tool" problem | ✅ **73% → ~3%** reduction |
| **Scaling law discovered** | ✅ **Smaller models benefit more** (+49 pts on 1.7B) |

**Key insight:** The hidden state right before a tool call is **linearly separable** from non-tool states — even in tiny models (1.7B). This geometric property is universal and exploitable, and **most valuable where text generation is weakest**.

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
| Qwen3-1.7B | ~3 GB | **Biggest gain: +49 pts** |
| Qwen3-4B | ~6 GB | **+42 pts gain** |
| Qwen3-8B | ~14 GB | Main reference, **+26 pts** |
| Qwen3-14B | ~28 GB | Optional, if budget allows |

Models are loaded and unloaded **sequentially**—only one in VRAM at a time.

---

## 📁 Output Structure

```
results/
├── sprint0/                          # Baseline & diagnostics
│   ├── scaling_analysis.json         # Geometric scaling across models
│   ├── qwen3-1.7b/                   # 1.7B results (baseline 26%)
│   ├── qwen3-4b/                     # 4B results (baseline 26%)
│   └── qwen3-8b/                     # 8B results (baseline 50%)
│
├── sprint1/                          # Â detector results
│   ├── qwen3-1.7b/                   # 75% success rate
│   ├── qwen3-4b/                     # 68% success rate
│   └── qwen3-8b/                     # 76% success rate
│
└── sprint_1_results/                  # Full analysis & benchmarks
    ├── scaling_comparison.md          # Cross-model analysis
    ├── qwen3-1.7b/evaluation.json
    ├── qwen3-4b/evaluation.json
    └── qwen3-8b/evaluation.json
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

## 📈 **Sprint 1 Results — Detailed by Model**

### Qwen3-1.7B (from 26% to 75%)
```
✅ Success rate: 75.0%
🎯 Correct tool usage: 75.0%
🔧 Optimal tool usage: 50.0%
❌ No-tool rate: 3.1%
🔥 Hallucinated tools: 9.4%
```

### Qwen3-4B (from 26% to 68%)
```
✅ Success rate: 67.7%
🎯 Correct tool usage: 61.3%
🔧 Optimal tool usage: 51.6%
❌ No-tool rate: 3.2%
🔥 Hallucinated tools: 9.7%
```

### Qwen3-8B (from 50% to 76%)
```
✅ Success rate: 76.3%
🎯 Correct tool usage: 78.9%
🔧 Optimal tool usage: 52.6%
❌ No-tool rate: 2.6%
🔥 Hallucinated tools: 2.6%
```

**Key achievement:** The "no_tool" failure mode — dominant in baseline (43-73%) — is now **almost eliminated across all scales** (down to ~3%).

[Full results →](sprint_1_results/)

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
