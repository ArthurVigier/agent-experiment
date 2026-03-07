# JEPA-Agent — Sprint 0

*An AI agent based on Joint Embedding Predictive Architecture.*
Sprint 0: baseline + diagnostics + Â validation + **geometric scaling analysis**.

> **Note:** All code comments and docstrings in the project are in **French** (a deliberate choice for maintainers), but this README and user-facing interfaces are in English.

---

## 🚀 Quick Start (RunPod A100 80GB)

```bash
# Setup
pip install -r requirements.txt
huggingface-cli login  # HF token for gated models (optional for Qwen)

# ─── Single model ───
python run_sprint0.py --model Qwen/Qwen3-8B

# ─── Multi-model scaling analysis (recommended) ───

# Preset: 1.5B + 3B + 7B (same family, isolates size effect)
python run_sprint0.py --scaling-preset

# Custom: choose your models
python run_sprint0.py --models Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B

# Maximum delta: add the 0.5B
python run_sprint0.py --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B

# ─── Options ───
# Quick test (single tasks only)
python run_sprint0.py --scaling-preset --tasks single

# With pre-computed R̂ for cos(Â, R̂) comparison
python run_sprint0.py --scaling-preset --r-hat-dir r_hat/

# Skip Â extraction (0A + 0B only)
python run_sprint0.py --scaling-preset --skip-0c
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
results/sprint0/
├── config.json                          # Run configuration
├── scaling_analysis.json                # Comparative analysis (Sprint 0D)
├── qwen2.5-1.5b-instruct/              # Per model
│   ├── traces.json                      # Complete traces
│   ├── failure_analysis.json            # Failure diagnosis (0B)
│   ├── a_hat_traces.json                # Â results (0C)
│   ├── a_hat_extracted.npy              # Extracted Â direction
│   ├── summary.json                     # Model summary
│   └── hidden_states/                   # .npy per step
├── qwen2.5-3b-instruct/
│   └── ...
└── qwen3-8b/
    └── ...
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
├── geometry/                  # Sprint 0C + Sprint 1 (coming soon)
├── run_sprint0.py            # Main entry point (multi-model)
├── evaluate_traces.py        # LLM-as-a-judge evaluation (more reliable than rule-based)
├── evaluate_deep.py          # Deep evaluation with custom judge models
└── requirements.txt
```

---

## 🤖 Evaluation with LLM-as-a-Judge

For more reliable and nuanced evaluation, we provide two additional scripts that use LLMs to judge agent performance:

### `evaluate_traces.py`

Uses an LLM (default: Qwen3-8B) to evaluate traces based on:
- Task completion
- Correct tool usage
- Answer quality
- Efficiency
- Reasoning quality

```bash
# Evaluate traces from a Sprint 0 run
python evaluate_traces.py \
    --traces results/sprint0/qwen3-8b/traces.json \
    --tasks baselines/tasks.py \
    --output results/sprint0/qwen3-8b/evaluation.json \
    --judge-model Qwen/Qwen3-8B
```

### `evaluate_deep.py`

A more thorough evaluation that includes:
- Step-by-step reasoning analysis
- Failure mode classification (wrong tool, wrong timing, loop, hallucination, etc.)
- Cross-validation with multiple judge models

```bash
# Deep evaluation with multiple judges
python evaluate_deep.py \
    --traces results/sprint0/qwen3-8b/traces.json \
    --tasks baselines/tasks.py \
    --output-dir results/sprint0/qwen3-8b/deep_eval/ \
    --judge-models Qwen/Qwen3-8B Qwen/Qwen3-4B
```

### Why LLM-as-a-Judge?

Rule-based evaluation (`failure_analysis.py`) is fast but limited:
- ✅ Good for clear-cut cases (no_tool, wrong_tool)
- ❌ Struggles with nuanced success criteria
- ❌ Can't assess answer quality or reasoning depth

LLM-based evaluation provides:
- ✅ Human-like judgment of answer correctness
- ✅ Detection of subtle failure modes (hallucinated tools, wrong timing)
- ✅ Reasoning quality assessment
- ✅ Cross-validation with multiple judge models

The results from `evaluate_traces.py` are saved in the same format as the rule-based analysis, making it easy to compare both approaches.

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
  author = {Your Name},
  title = {JEPA-Agent: Geometric Agency for LLM Tool Use},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/jepa-agent}
}
```

---

## 📜 License

Apache 2.0 — free for academic and commercial use, with attribution.
