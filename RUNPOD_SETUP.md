
RunPod Setup — Sprint 0 JEPA-Agent
A practical, step-by-step guide. From RunPod account to first results.

Note: All code comments and docstrings in the project are in French, but this guide and the README are in English for accessibility.

1. Hardware Selection
Which GPU?
Your choice depends on what you want to test:

Scenario	GPU	VRAM	Cost/h	Sprint 0 Duration	Total Budget
Single model (7B)	1× A100 80GB SXM	80 GB	~$1.64	2-3h	$3-5
Scaling preset (1.5B+3B+7B)	1× A100 80GB SXM	80 GB	~$1.64	4-6h	$7-10
Full scaling (0.5B+1.5B+3B+7B)	1× A100 80GB SXM	80 GB	~$1.64	5-8h	$8-13
Quick test	1× RTX 4090	24 GB	~$0.44	1-2h (single only)	$1-2
Minimal budget	1× L40 / A40	48 GB	~$0.80	3-5h	$3-4
Recommendation: A100 80GB SXM, On-Demand. It's the sweet spot. 80GB VRAM means you load the 7B in bf16 (15GB), leaving 65GB for hidden states in memory and any peaks. It's also enough for models up to 32B if you want to push scaling later.

Why On-Demand, not Spot? Sprint 0 takes 2-6h depending on the number of models. A preempted Spot instance interrupts the run, losing in-progress traces and forcing a full restart. The savings (~30%) aren't worth the risk for a run of this duration.

VRAM per model (bf16)
Model	bf16 VRAM	GPTQ-4bit VRAM	Note
Qwen3-0.6B	~1.2 GB	~0.5 GB	Baseline, trivial
Qwen3-1.7B	~3.5 GB	~1.5 GB	Light
Qwen3-4B	~7 GB	~3 GB	Comfortable
Qwen3-8B	~15 GB	~5 GB	Reference
Qwen3-14B	~29 GB	~10 GB	Optional, large
Qwen3-32B	~65 GB	~20 GB	Tight on A100 80GB
Models are loaded and unloaded sequentially—only one in VRAM at a time. 0.5B + 1.5B + 3B + 7B ≠ 26GB in parallel, but a max of 15GB at any time (the 7B is the largest).

Complete pod configuration
Parameter	Value	Why
GPU	1× NVIDIA A100 80GB SXM	See table above
GPU Count	1	Enough for Sprint 0
Template	runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04	PyTorch + CUDA pre-installed
Container Disk	50 GB	Space for pip, transformers cache
Volume Disk	100 GB	HF models (~15GB/model) + hidden states (~5-15GB) + margin
Volume Mount	/workspace	PERSISTENT between stop/restart
vCPU	8+	Preprocessing, not critical
RAM	62 GB (or max available)	Accumulated hidden states fit in RAM
Type	On-Demand	No Spot—preempt = lost run
The 100GB volume: it's larger than the 50GB in previous guides. In scaling mode with 4 models, hidden states + cached HF models take ~40-60GB. 100GB gives headroom.

2. Pod Creation — Step by Step
2a. Prerequisites
RunPod account: https://www.runpod.io/

Credit added ($25 minimum recommended for Sprint 0 + margin)

SSH Key: Settings → SSH Keys → add your public key (cat ~/.ssh/id_rsa.pub)

2b. Deploy
Dashboard → Pods → + Deploy

Select GPU Pod

Filter: check "80 GB" VRAM, sort by price

Select A100 80GB SXM (the cheapest available)

Customize Deployment:

Template: search runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

Container Disk: 50 GB

Volume Disk: 100 GB

Volume Mount Path: /workspace

Deploy On-Demand

Wait ~1-3 min for the pod to show "Running"

2c. Connect
SSH (recommended):

bash
# The command is shown in the pod dashboard
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
Web Terminal: Dashboard → click the pod → "Connect" → "Start Web Terminal"

Jupyter: The PyTorch template also exposes Jupyter Lab on port 8888. Useful for interactively inspecting results.

3. Environment Setup
Everything happens in /workspace (persistent volume). Do NOT work in /root (container disk, non-persistent).

bash
# ═══ Move to persistent volume ═══
cd /workspace

# ═══ Upload the project ═══

# Option A: scp from your local machine
# (on your machine, NOT on the pod)
# scp -P <port> jepa-agent-sprint0.tar.gz root@<pod-ip>:/workspace/

# Option B: wget if hosted somewhere
# wget <url>/jepa-agent-sprint0.tar.gz

# Extract
tar xzf jepa-agent-sprint0.tar.gz
cd jepa-agent

# ═══ Install dependencies ═══
pip install -r requirements.txt

# Verify installation
pip list | grep -E "torch|transformers|accelerate|duckduckgo|sklearn"
# Expected:
# torch                    2.x.x
# transformers             4.4x.x
# accelerate               0.x.x
# duckduckgo-search        6.x.x
# scikit-learn             1.x.x
HuggingFace login
Qwen3 models are under Apache 2.0, not gated—no login needed for them. But if you want to test gated models (Llama 3, Mistral), you'll need a token:

bash
pip install huggingface_hub
huggingface-cli login
# → Paste your token from https://huggingface.co/settings/tokens
# → Accept model terms on the HF page
Verify CUDA and GPU
bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1e9:.1f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
"
# Expected:
# PyTorch: 2.4.x
# CUDA available: True
# GPU: NVIDIA A100-SXM4-80GB
# VRAM: 80.0 GB
# Compute capability: 8.0
Prepare the file sandbox
File tools (read_file, csv_analyze, etc.) look in /tmp/jepa_sandbox/. You need to create realistic files there for tasks to work:

bash
mkdir -p /tmp/jepa_sandbox/data

# CSV with real data
cat > /tmp/jepa_sandbox/results.csv << 'EOF'
accuracy,loss,epoch,lr,model
0.823,0.45,1,0.001,baseline
0.856,0.39,2,0.001,baseline
0.871,0.35,3,0.001,baseline
0.889,0.31,4,0.001,baseline
0.902,0.28,5,0.001,baseline
0.847,0.42,1,0.001,experiment_a
0.878,0.36,2,0.001,experiment_a
0.891,0.32,3,0.001,experiment_a
0.912,0.27,4,0.001,experiment_a
0.925,0.24,5,0.001,experiment_a
EOF

cp /tmp/jepa_sandbox/results.csv /tmp/jepa_sandbox/metrics.csv
cp /tmp/jepa_sandbox/results.csv /tmp/jepa_sandbox/data.csv
cp /tmp/jepa_sandbox/results.csv /tmp/jepa_sandbox/data/report.csv

# Text file
cat > /tmp/jepa_sandbox/report.txt << 'EOF'
This is a test report about the Q3 performance of the engineering team.
Key findings: deployment frequency increased by 23%, mean time to recovery
decreased by 15%, and customer satisfaction scores improved across all segments.
The team successfully migrated 3 critical services to the new infrastructure.
EOF
cp /tmp/jepa_sandbox/report.txt /tmp/jepa_sandbox/data/report.txt

# JSON
cat > /tmp/jepa_sandbox/users.json << 'EOF'
{"users": [
  {"id": 1, "name": "Alice", "role": "admin", "email": "alice@example.com"},
  {"id": 2, "name": "Bob", "role": "developer", "email": "bob@example.com"},
  {"id": 3, "name": "Charlie", "role": "admin", "email": "charlie@example.com"},
  {"id": 4, "name": "Diana", "role": "designer", "email": "diana@example.com"}
]}
EOF

cat > /tmp/jepa_sandbox/config.json << 'EOF'
{"model": "qwen2.5-7b", "learning_rate": 0.001, "batch_size": 32, "epochs": 10}
EOF

# Python script
cat > /tmp/jepa_sandbox/test_script.py << 'EOF'
def main():
    print('hello from test file')

if __name__ == '__main__':
    main()
EOF
Complete verification
bash
python -c "
import sys; sys.path.insert(0, '.')
from baselines.tools import TOOLS, TOOL_MAP
from baselines.tasks import ALL_TASKS, task_stats
from baselines.executors import EXECUTOR_MAP, SANDBOX_DIR
import os

print(f'Tools: {len(TOOLS)}')
print(f'Tasks: {task_stats()}')
print(f'Executors: {len(EXECUTOR_MAP)}')
print(f'Sandbox: {SANDBOX_DIR}')
sandbox_files = list(SANDBOX_DIR.iterdir()) if SANDBOX_DIR.exists() else []
print(f'Sandbox files: {len(sandbox_files)}')
for f in sandbox_files[:10]:
    print(f'  {f.name} ({f.stat().st_size}B)')

# Test a real executor
result = EXECUTOR_MAP['calculator']('2**32')
print(f'Calculator test: 2^32 = {result}')
result = EXECUTOR_MAP['read_file']('results.csv')
print(f'Read file test: {result[:80]}...')
print()
print('✓ READY TO GO')
"
If you see ✓ READY TO GO, everything is set.

4. Running Sprint 0
Step 1: Quick test (5-10 min)
Always run a quick test first. This validates that the model loads, the agent parses tool calls, and the executors run.

bash
python run_sprint0.py \
    --model Qwen/Qwen3-8B \
    --tasks single \
    --max-steps 5 \
    --output-dir results/sprint0_test \
    --skip-0c
What should happen:

Model loads (~1 min, first time ~5 min for download)

Sprint 0-PRE runs in < 30 seconds → displays embedding silhouette

Sprint 0A runs 40 single tasks → each task shows ✓ or ✗ with timing

Sprint 0B displays the failure report

Warning signs:

OOM on load → see Troubleshooting

0 tool calls on the first 10 tasks → model isn't parsing the ReAct format → see Troubleshooting

All tasks < 2 seconds → model generates empty responses

Step 2: Full single-model run (2-3h)
bash
python run_sprint0.py \
    --model Qwen/Qwen3-8B \
    --output-dir results/sprint0
Step 3: Full multi-model scaling run (4-6h)
bash
# Preset: 1.5B + 3B + 7B (recommended)
python run_sprint0.py \
    --scaling-preset \
    --output-dir results/sprint0

# OR with 0.5B for max delta (5-8h)
python run_sprint0.py \
    --models \
        Qwen/Qwen3-0.6B \
        Qwen/Qwen3-1.7B \
        Qwen/Qwen3-4B \
        Qwen/Qwen3-8B \
    --output-dir results/sprint0
Step 4: With pre-computed R̂ (optional)
If you have R̂ from your register-geometry paper:

bash
# Upload R̂ from your machine
mkdir -p /workspace/jepa-agent/r_hat
# scp -P <port> r_hat_*.npy root@<pod-ip>:/workspace/jepa-agent/r_hat/
# Name files: qwen2.5-7b-instruct.npy, qwen2.5-3b-instruct.npy, etc.

python run_sprint0.py \
    --scaling-preset \
    --r-hat-dir r_hat/ \
    --output-dir results/sprint0
Running in background (recommended for long runs)
bash
# With nohup—the run continues even if you disconnect
nohup python run_sprint0.py \
    --scaling-preset \
    --output-dir results/sprint0 \
    > sprint0_stdout.log 2>&1 &

# Get the PID
echo $!
# → 12345

# Check that it's running
ps aux | grep run_sprint0
Monitoring during execution
bash
# Terminal 1: Real-time VRAM
watch -n 5 nvidia-smi

# Terminal 2: Logs
tail -f sprint0.log

# Terminal 3: Progress
watch -n 30 'ls -la results/sprint0/*/traces.json 2>/dev/null'
5. Interpreting Quick Results
Sprint 0-PRE (immediate, during loading)
text
SPRINT 0-PRE : STRUCTURE FONCTIONNELLE DANS LES EMBEDDINGS
═══════════════════════════════════════════════════════════
  Silhouette moyenne : 0.XXX
> 0.30: excellent. The functional space exists in the weights. Geometric routing has a solid foundation.

0.10 - 0.30: partial. There's signal but it's noisy. Usable as initialization.

< 0.10: no structure in embeddings. Normal for some models—doesn't mean hidden states lack structure.

Sprint 0B (after 0A, the key diagnostic)
text
Mode d'échec dominant : wrong_tool
This determines Sprint 1. Look at the percentage:

wrong_tool > 30% → Sprint 1A (geometric routing)

loop/no_tool > 30% → Sprint 1B (Â detector)

premature_stop > 30% → Sprint 1C (trajectory planner)

Sprint 0C (if not --skip-0c)
text
AUC(R̂ → tool call) = 0.XXX
> 0.75: R̂ predicts tool calls. Â detector initializable with R̂.

0.60 - 0.75: partial signal. Supervised probe needed.

< 0.60: fallback cascade triggered automatically.

Sprint 0D (multi-model only)
The comparison table and scaling verdict. This could be the most important result of the entire Sprint 0.

6. Retrieving Results
Output structure
text
results/sprint0/
├── config.json                          # Run config
├── scaling_analysis.json                # Sprint 0D (if multi-model)
│
├── qwen2.5-0.5b-instruct/              # Per model
│   ├── embedding_structure.json         # Sprint 0-PRE ← SMALL, IMPORTANT
│   ├── tool_embeddings.npy              # Embeddings for reuse
│   ├── tool_similarity_matrix.npy       # Similarity matrix
│   ├── traces.json                      # Sprint 0A ← MEDIUM, CRITICAL
│   ├── failure_analysis.json            # Sprint 0B ← SMALL, CRITICAL
│   ├── a_hat_traces.json                # Sprint 0C ← SMALL, IMPORTANT
│   ├── a_hat_extracted.npy              # Â direction
│   ├── summary.json                     # Summary
│   └── hidden_states/                   # ← LARGE (2-10GB per model)
│       ├── s01_step000.npy
│       └── ...
│
├── qwen2.5-1.5b-instruct/
│   └── ...
├── qwen2.5-3b-instruct/
│   └── ...
└── qwen2.5-7b-instruct/
    └── ...
Download (in priority order)
bash
# 1. JSONs first (a few KB, the essential results)
cd /workspace/jepa-agent
for dir in results/sprint0/*/; do
    echo "=== $dir ==="
    ls -lh "$dir"/*.json 2>/dev/null
done

# Compress only JSONs
find results/sprint0 -name "*.json" | tar czf /workspace/sprint0_json.tar.gz -T -

# 2. .npy (embeddings + extracted Â, a few MB)
find results/sprint0 -name "*.npy" ! -path "*/hidden_states/*" | tar czf /workspace/sprint0_npy.tar.gz -T -

# 3. Hidden states (LARGE, 2-10GB per model)
# Only if you want to run Sprint 0C-bis or Sprint 1 locally
# Otherwise, keep them on the RunPod volume
tar czf /workspace/sprint0_hidden_states.tar.gz results/sprint0/*/hidden_states/
bash
# On your local machine:
scp -P <port> root@<pod-ip>:/workspace/sprint0_json.tar.gz .
scp -P <port> root@<pod-ip>:/workspace/sprint0_npy.tar.gz .
# Hidden states only if necessary (LARGE transfer)
7. After Sprint 0
STOP the pod, do NOT TERMINATE it
Stop: GPU released, no GPU cost. Persistent volume (~$0.10/GB/month ≈ $10/month for 100GB)

Terminate: EVERYTHING is destroyed. Downloaded models, hidden states, everything.

The volume contains:

HuggingFace cache (~/.cache/huggingface/): ~15-60GB depending on downloaded models

Hidden states: ~5-30GB depending on number of models

Results: a few MB

You'll restart the same pod for Sprint 1. Everything will still be there.

Optional cleanup (reduce volume cost)
bash
# Delete hidden states for models you're not interested in
rm -rf results/sprint0/qwen2.5-0.5b-instruct/hidden_states/

# Delete cache of unloaded models (re-download needed)
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/

# Check space
du -sh /workspace/* | sort -h
8. Troubleshooting
OOM (Out of Memory)
text
torch.cuda.OutOfMemoryError: CUDA out of memory
Causes and solutions:

Another model still in VRAM—garbage collector didn't release

bash
# Check
nvidia-smi
# If VRAM > 0 with no process: restart Python kernel
Model too large—7B in bf16 = 15GB, fits easily on 24GB+

bash
# Use quantized version
pip install auto-gptq optimum
python run_sprint0.py --model Qwen/Qwen3-8B-GPTQ-Int4
Hidden states accumulating in VRAM—detach()ed tensors should be on CPU

bash
# Check in a Python shell
import torch
print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
Model makes no tool calls
Symptom: all tasks end in max_steps_reached, 0 tool calls in traces.

Quick diagnostic:

python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda"
)

messages = [
    {"role": "system", "content": "You have tools. To use one, write: Action: web_search(query=\"your query\")"},
    {"role": "user", "content": "What is the population of France?"}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
If output doesn't contain Action: web_search(...):

The model isn't following the ReAct format → try a larger model (1.5B is often too small for complex instructions)

Reduce system prompt (the current prompt with 25 tool descriptions is long)

Increase temperature to 0.3

DuckDuckGo rate limited
text
duckduckgo_search.exceptions.RatelimitException
Solution: add a delay between requests. In baselines/executors.py, add time.sleep(2) at the start of api_web_search. Or ignore (the fallback returns an error message, agent continues).

Sandbox files not found
text
ERROR: File not found: 'results.csv'
The sandbox is in /tmp/jepa_sandbox/. Check that step 3 (sandbox preparation) was executed. /tmp is cleared on pod restart—re-run the sandbox setup on every restart.

Run crashed mid-way
Traces are saved incrementally in traces.json. If the run crashes at step 45/60, the first 44 traces are saved. You can:

bash
# See how many traces were saved
python -c "import json; d=json.load(open('results/sprint0/qwen2.5-7b-instruct/traces.json')); print(f'{len(d)} traces')"

# Re-run only missing tasks (not yet implemented, but existing traces are usable)
# Sprint 0B and 0C work on partial traces
9. Pre-launch Checklist
text
PREREQUISITES
  [ ] RunPod account with $25+ credit
  [ ] SSH key added in RunPod Settings

DEPLOYMENT
  [ ] A100 80GB SXM pod deployed (On-Demand)
  [ ] Status: Running
  [ ] SSH connected

SETUP
  [ ] cd /workspace
  [ ] Project extracted in /workspace/jepa-agent
  [ ] pip install -r requirements.txt
  [ ] nvidia-smi → GPU visible
  [ ] torch.cuda.is_available() → True
  [ ] Sandbox files created (/tmp/jepa_sandbox/)
  [ ] Complete verification: "✓ READY TO GO"

QUICK TEST
  [ ] python run_sprint0.py --model Qwen/Qwen3-8B --tasks single --max-steps 5 --skip-0c
  [ ] Sprint 0-PRE displays silhouette
  [ ] Agent makes tool calls (not 0 tool calls everywhere)
  [ ] No OOM

LAUNCH
  [ ] nohup python run_sprint0.py --scaling-preset --output-dir results/sprint0 > stdout.log 2>&1 &
  [ ] watch nvidia-smi → GPU is working
  [ ] tail -f sprint0.log → tasks are progressing

AFTER
  [ ] JSON results downloaded
  [ ] Pod STOPPED (not terminated)
  [ ] Results inspected locally
10. Detailed Time Estimates
Single model (Qwen3-8B)
Step	Duration	Detail
Model loading	1-5 min	1 min if cached, 5 min first download
0-PRE	< 30 sec	Embedding lookup, no forward pass
0A (60 tasks)	1.5-3h	~1-3 min/task (depends on steps)
0B	< 1 min	Trace analysis, no GPU
0C	2-5 min	Projections on already saved hidden states
Total	~2-3h
Scaling preset (1.5B + 3B + 7B)
Step	Duration	Detail
Qwen3-1.7B	30-60 min	Small model, fast generation
Unload + load 3B	2-3 min
Qwen3-4B	45-90 min
Unload + load 7B	2-5 min
Qwen3-8B	1.5-3h	Slowest
0D (comparative analysis)	< 1 min
Total	~3.5-6h
Full scaling (0.5B + 1.5B + 3B + 7B)
Add ~20-40 min for the 0.5B. Total: ~4-7h.
