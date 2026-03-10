#!/usr/bin/env python3
"""
run_sprint2.py — Sprint 2: Geometric Memory (Scar Buffer)

<<<<<<<< HEAD:scripts/run_sprint2.py
Integrates action predictor and scar buffer into the agent,
and benchmarks against Sprint 1 baseline.

Usage:
  python run_sprint2.py --sprint1-dir results/sprint1 --model Qwen/Qwen3-8B
========
Supports Qwen 3.5 models with MoE architecture.
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from run_sprint1 (original version)
from run_sprint1 import (
    AHatDetector,
    Sprint1Agent,
    HiddenStateLogger,
    model_short_name,
    SCALING_MODELS,
)

# Import memory modules
from memory.scar_buffer import ScarBuffer
<<<<<<<< HEAD:scripts/run_sprint2.py
from memory.predictor_moe import MoEAwarePredictor
from memory.integrator_moe import MoEGeometricIntegrator

# Import baselines
========
from memory.predictor import ActionPredictor
from memory.integrator import GeometricIntegrator
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
from baselines.tools import TOOL_MAP, get_tool_names
from baselines.tasks import ALL_TASKS, SINGLE_TASKS, CHAIN_TASKS, ADVERSARIAL_TASKS
from baselines.react_agent import ReActAgent, HiddenStateLogger
from baselines.failure_analysis import analyze_traces

<<<<<<<< HEAD:scripts/run_sprint2.py
========
# Import from run_sprint1
from scripts.run_sprint1 import (
    Sprint1Agent, SavingHiddenStateLogger, SystemMonitor,
    force_cleanup, AHatDetector
)

>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("sprint2.log")],
)
logger = logging.getLogger(__name__)


# ============================================================================
<<<<<<<< HEAD:scripts/run_sprint2.py
# System monitoring and cleanup (not in original run_sprint1)
# ============================================================================

class SystemMonitor:
    """Monitor RAM/VRAM in real time."""

    def __init__(self, log_interval: int = 30):
        self.log_interval = log_interval
        self.last_log = 0
        self.peak_ram = 0
        self.peak_vram = 0

    def check(self, force: bool = False) -> dict:
        now = time.time()
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / 1e9
        ram_percent = ram.percent
        self.peak_ram = max(self.peak_ram, ram_used_gb)

        vram_used_gb = 0
        vram_percent = 0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated()
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_used_gb = vram_used / 1e9
            vram_percent = 100 * vram_used / vram_total
            self.peak_vram = max(self.peak_vram, vram_used_gb)

        if force or (now - self.last_log > self.log_interval):
            logger.info(
                f"📊 RAM: {ram_used_gb:.1f}GB ({ram_percent:.1f}%) | "
                f"VRAM: {vram_used_gb:.1f}GB ({vram_percent:.1f}%) | "
                f"Peak RAM: {self.peak_ram:.1f}GB | Peak VRAM: {self.peak_vram:.1f}GB"
            )
            self.last_log = now

        if ram_percent > 90:
            logger.warning(f"⚠️  RAM critical: {ram_percent:.1f}%")

        return {
            "ram_used_gb": ram_used_gb,
            "ram_percent": ram_percent,
            "vram_used_gb": vram_used_gb,
            "vram_percent": vram_percent,
            "peak_ram": self.peak_ram,
            "peak_vram": self.peak_vram,
        }


def force_cleanup():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# Saving hidden state logger (to avoid RAM accumulation)
# ============================================================================

class SavingHiddenStateLogger(HiddenStateLogger):
    """Variant that immediately saves hidden states to disk."""

    def get_last_hidden_state(self, attention_mask=None):
        h = super().get_last_hidden_state(attention_mask)
        # h is already a numpy array in HiddenStateLogger
        return h

    def save(self, task_id: str, step_idx: int, h: np.ndarray) -> str:
        filename = f"{task_id}_step{step_idx:03d}.npy"
        path = self.output_dir / filename
        np.save(path, h)
        return str(path)


# ============================================================================
# Sprint 2 Agent (extending Sprint1Agent with memory)
# ============================================================================

========
# Qwen 3.5 model mapping
# ============================================================================

QWEN35_MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B",      # fallback
    "1.5b": "Qwen/Qwen2.5-1.5B",      # fallback
    "3b": "Qwen/Qwen2.5-3B",          # fallback
    "7b": "Qwen/Qwen2.5-7B",          # fallback
    "14b": "Qwen/Qwen2.5-14B",        # fallback
    "32b": "Qwen/Qwen2.5-32B",        # fallback
    "35b": "Qwen/Qwen3.5-35B-A3B",    # MoE, 3B activated
    "72b": "Qwen/Qwen3.5-72B",        # dense
    "122b": "Qwen/Qwen3.5-122B-A10B", # MoE, 10B activated
}


# ============================================================================
# Sprint 2 Agent
# ============================================================================

>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
class Sprint2Agent(Sprint1Agent):
    """Agent with Â gating + geometric memory (scar buffer)."""

    def __init__(
        self,
        model,
        tokenizer,
        tools,
        a_detector,
        integrator: GeometricIntegrator,
        hs_logger,
        max_steps: int = 10,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        device: str = "cuda",
<<<<<<<< HEAD:scripts/run_sprint2.py
        capture_experts: bool = True,
========
        capture_experts: bool = False,
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            tools=tools,
            a_detector=a_detector,
            tool_router=None,          # not used in sprint2
            logits_router=None,         # not used in sprint2
            hs_logger=hs_logger,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            routing_mode="none",        # we handle forcing ourselves
            a_hat_enabled=True,
        )
        self.integrator = integrator
        self.capture_experts = capture_experts
        self.last_expert_routing = None

    def _generate(self, messages):
<<<<<<<< HEAD:scripts/run_sprint2.py
        """Override to capture expert routing if possible."""
========
        """Override to optionally capture expert routing."""
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(
                f"{'System' if m['role']=='system' else 'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in messages
            ) + "\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        t0 = time.perf_counter()

<<<<<<<< HEAD:scripts/run_sprint2.py
========
        # Generation kwargs
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.temperature > 0 else None,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

<<<<<<<< HEAD:scripts/run_sprint2.py
========
        # Add expert routing capture if supported
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
        if self.capture_experts:
            try:
                generate_kwargs["output_router_logits"] = True
            except:
                logger.warning("output_router_logits not supported")
                self.capture_experts = False

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        gen_time = (time.perf_counter() - t0) * 1000

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0, input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

<<<<<<<< HEAD:scripts/run_sprint2.py
========
        # Logits
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
        logits_last = None
        if hasattr(outputs, 'scores') and outputs.scores:
            logits_last = outputs.scores[-1][0]

<<<<<<<< HEAD:scripts/run_sprint2.py
========
        # Expert routing
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
        expert_routing = None
        if self.capture_experts and hasattr(outputs, 'router_logits') and outputs.router_logits:
            expert_routing = outputs.router_logits[-1][0].cpu().numpy()
            self.last_expert_routing = expert_routing

        # Hidden state
        h = self.hs_logger.get_last_hidden_state(inputs.get("attention_mask"))

        return text, gen_time, h, logits_last, expert_routing

    def run(self, task_id: str, task_prompt: str, task_category: str = "",
            expected_tools: list = None):
        from baselines.react_agent import Trace, StepRecord

        trace = Trace(
            task_id=task_id,
            task_prompt=task_prompt,
            task_category=task_category,
            expected_tools=expected_tools or [],
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        t_start = time.perf_counter()
        expert_routings = []

        for step_idx in range(self.max_steps):
            text, gen_time_ms, h, logits_last, expert_routing = self._generate(messages)
            expert_routings.append(expert_routing)

            hs_path = None
            if h is not None:
                hs_path = self.hs_logger.save(task_id, step_idx, h)

            # Â detection
            a_hat_triggered = False
            a_hat_confidence = 0.0
            if self.a_detector is not None and h is not None:
                should_call, confidence = self.a_detector.predict(h)
                a_hat_confidence = confidence
                if should_call:
                    a_hat_triggered = True
                    logger.info(f"🔥 Â TRIGGERED! confidence={confidence:.2f}")

            thought = self._parse_thought(text)
            final_answer = self._parse_final_answer(text)
            tool_name_text, params_text = self._parse_action(text)

            # Decision: which tool to use?
            actual_tool = None
            actual_params = None
            routing_source = "none"

<<<<<<<< HEAD:scripts/run_sprint2.py
========
            # 1. If agent generated a valid tool call
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
            if tool_name_text and tool_name_text in self.tools:
                actual_tool = tool_name_text
                actual_params = params_text
                routing_source = "textual"
                logger.info(f"📝 Textual tool call: {actual_tool}")

<<<<<<<< HEAD:scripts/run_sprint2.py
========
            # 2. If Â triggered and no final answer
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
            elif a_hat_triggered and not final_answer:
                # Fallback forced tool (no routing for now)
                actual_tool = "web_search"
                actual_params = {"query": "default search"}
                routing_source = "forced"
                logger.info(f"🔥 Â FORCED tool call: {actual_tool} (confidence={a_hat_confidence:.2f})")

<<<<<<<< HEAD:scripts/run_sprint2.py
========
            # 3. If final answer without forced tool
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
            if final_answer and actual_tool is None:
                step = StepRecord(
                    step_idx=step_idx, input_text=messages[-1]["content"],
                    thought=thought, action=None, action_params=None,
                    observation=None, generated_text=text,
                    hidden_state_path=hs_path, hidden_state_layer=self.hs_logger.mid_layer,
                    generation_time_ms=gen_time_ms, â_score=float(a_hat_confidence)
                )
                trace.steps.append(step)
                trace.final_answer = final_answer
                break

<<<<<<<< HEAD:scripts/run_sprint2.py
========
            # 4. Execute tool if we have one
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
            if actual_tool:
                # Tool embedding (one-hot)
                tool_names = get_tool_names()
                z_tool = np.zeros(len(tool_names))
                if actual_tool in tool_names:
                    z_tool[tool_names.index(actual_tool)] = 1.0

                # Predict next state with scar correction
<<<<<<<< HEAD:scripts/run_sprint2.py
                h_pred, corrections, expert_pred = self.integrator.predict(
========
                h_pred, corrections = self.integrator.predict(
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
                    h_before=h,
                    z_tool=z_tool,
                    tool_name=actual_tool,
                    expert_routing=expert_routing,
                )

<<<<<<<< HEAD:scripts/run_sprint2.py
========
                # Execute
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
                t_exec = time.perf_counter()
                try:
                    observation = self.tools[actual_tool].execute(**actual_params)
                except Exception as e:
                    observation = f"ERROR: {e}"
                exec_time = (time.perf_counter() - t_exec) * 1000

                step = StepRecord(
                    step_idx=step_idx, input_text=messages[-1]["content"],
                    thought=thought, action=actual_tool, action_params=actual_params,
                    observation=observation, generated_text=text,
                    hidden_state_path=hs_path, hidden_state_layer=self.hs_logger.mid_layer,
                    generation_time_ms=gen_time_ms, tool_execution_time_ms=exec_time,
                    â_score=float(a_hat_confidence)
                )
                trace.steps.append(step)
                trace.num_tool_calls += 1
                trace.tools_used.append(actual_tool)

                # Add to context
                if routing_source != "textual":
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content":
                        f"[Geometric routing selected tool '{actual_tool}']\n"
                        f"Observation: {observation}"})
                else:
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})

            else:
                # No tool, no final answer → force continuation
                step = StepRecord(
                    step_idx=step_idx, input_text=messages[-1]["content"],
                    thought=thought, action=None, action_params=None,
                    observation=None, generated_text=text,
                    hidden_state_path=hs_path, hidden_state_layer=self.hs_logger.mid_layer,
                    generation_time_ms=gen_time_ms, â_score=float(a_hat_confidence)
                )
                trace.steps.append(step)

                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content":
                    "Please continue. Use a tool or provide a Final Answer."})

<<<<<<<< HEAD:scripts/run_sprint2.py
            # Learn from previous step if possible
========
            # Learn from previous step
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
            if step_idx > 0 and trace.steps[-2].action is not None:
                prev_step = trace.steps[-2]
                if prev_step.hidden_state_path and Path(prev_step.hidden_state_path).exists():
                    h_before = np.load(prev_step.hidden_state_path)

<<<<<<<< HEAD:scripts/run_sprint2.py
========
                    # Tool embedding for previous tool
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
                    tool_names = get_tool_names()
                    z_tool = np.zeros(len(tool_names))
                    if prev_step.action in tool_names:
                        z_tool[tool_names.index(prev_step.action)] = 1.0

<<<<<<<< HEAD:scripts/run_sprint2.py
                    exp_before = expert_routings[step_idx-1] if step_idx-1 < len(expert_routings) else None
                    exp_after = expert_routing

                    self.integrator.learn(
                        h_before=h_before,
                        z_tool=z_tool,
                        tool_name=prev_step.action,
                        h_predicted=None,
                        h_actual=h,
                        expert_routing_before=exp_before,
                        expert_routing_after=exp_after,
                        task_id=task_id,
                    )
========
                    # We don't have the prediction, but we can still learn from error
                    # This would need h_pred stored - for now, skip learning
                    pass
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py

            self.integrator.step()

        trace.total_time_ms = (time.perf_counter() - t_start) * 1000
        if trace.final_answer is None:
            trace.error = "max_steps_reached"
        return trace


# ============================================================================
# Training utilities
# ============================================================================

def train_predictor_from_traces(
    traces_path: Path,
    hidden_states_dir: Path,
    tool_names: list[str],
    hidden_dim: int,
    embed_dim: int = 128,
    num_experts: Optional[int] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
<<<<<<<< HEAD:scripts/run_sprint2.py
) -> MoEAwarePredictor:
    """Train MoEAwarePredictor on Sprint 1 traces."""
========
) -> ActionPredictor:
    """Train ActionPredictor on Sprint 1 traces."""
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    import torch.optim as optim
    import torch.nn as nn

    with open(traces_path) as f:
        traces = json.load(f)

<<<<<<<< HEAD:scripts/run_sprint2.py
========
    # Collect (h_before, z_tool, h_after) triples
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    X_h = []
    X_z = []
    Y = []

    for trace in traces:
        for i, step in enumerate(trace["steps"]):
            if step["action"] is None:
                continue
            if i + 1 >= len(trace["steps"]):
                continue

            hs_path = step.get("hidden_state_path")
            if not hs_path or not Path(hs_path).exists():
                continue
            h_before = np.load(hidden_states_dir / Path(hs_path).name)

<<<<<<<< HEAD:scripts/run_sprint2.py
========
            # h_after (next step)
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
            next_step = trace["steps"][i + 1]
            next_hs_path = next_step.get("hidden_state_path")
            if not next_hs_path or not Path(next_hs_path).exists():
                continue
            h_after = np.load(hidden_states_dir / Path(next_hs_path).name)

            z_tool = np.zeros(len(tool_names))
            if step["action"] in tool_names:
                z_tool[tool_names.index(step["action"])] = 1.0

<<<<<<<< HEAD:scripts/run_sprint2.py
            # Fake expert distributions (since we don't have real ones)
            exp_before = np.random.dirichlet(np.ones(num_experts))
            exp_after = np.random.dirichlet(np.ones(num_experts))

========
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
            X_h.append(h_before)
            X_z.append(z_tool)
            Y.append(h_after)

    if len(X_h) < 10:
<<<<<<<< HEAD:scripts/run_sprint2.py
        raise ValueError(f"Insufficient training data: {len(X_h)} samples")

    logger.info(f"Training MoE predictor on {len(X_h)} samples")

========
        raise ValueError(f"Not enough training data: {len(X_h)} samples")

    logger.info(f"Training predictor on {len(X_h)} samples")

    # Convert to tensors
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    X_h_t = torch.tensor(np.stack(X_h), dtype=torch.float32).to(device)
    X_z_t = torch.tensor(np.stack(X_z), dtype=torch.float32).to(device)
    Y_t = torch.tensor(np.stack(Y), dtype=torch.float32).to(device)

<<<<<<<< HEAD:scripts/run_sprint2.py
    predictor = MoEAwarePredictor(
========
    # Initialize predictor
    predictor = ActionPredictor(
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_experts=num_experts,
    ).to(device)

    optimizer = optim.AdamW(predictor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    batch_size = 32
<<<<<<<< HEAD:scripts/run_sprint2.py
========

>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    for epoch in range(epochs):
        perm = torch.randperm(len(X_h))
        epoch_loss = 0.0

        for i in range(0, len(X_h), batch_size):
            idx = perm[i:i+batch_size]
            h_batch = X_h_t[idx]
            z_batch = X_z_t[idx]
            y_batch = Y_t[idx]

            optimizer.zero_grad()
<<<<<<<< HEAD:scripts/run_sprint2.py
            h_pred, exp_pred = predictor(h_batch, z_batch, exp_batch)
            loss_h = criterion_h(h_pred, y_batch)
            loss_exp = criterion_exp(
                torch.log_softmax(exp_pred, dim=-1),
                torch.softmax(y_exp_batch, dim=-1)
            )
            loss = loss_h + 0.1 * loss_exp
========
            h_pred, _ = predictor(h_batch, z_batch)
            loss = criterion(h_pred, y_batch)
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        epoch_loss /= len(X_h)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.6f}")

    return predictor


# ============================================================================
# Main runner
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Sprint 2 — Geometric Memory")
<<<<<<<< HEAD:scripts/run_sprint2.py
    parser.add_argument("--sprint1-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
========
    parser.add_argument("--sprint1-dir", type=str, required=True,
                        help="Directory with Sprint 1 results")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model ID or size key (35b, 72b, 122b for Qwen 3.5)")
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    parser.add_argument("--tasks", type=str, default="single",
                        choices=["single", "chain", "adversarial", "all"])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/sprint2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--predictor-epochs", type=int, default=50)
    parser.add_argument("--skip-training", action="store_true")
<<<<<<<< HEAD:scripts/run_sprint2.py
    parser.add_argument("--num-experts", type=int, default=8)
========
    parser.add_argument("--capture-experts", action="store_true",
                        help="Capture expert routing (for Qwen 3.5 MoE models)")
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    return parser.parse_args()


def main():
    args = parse_args()
    sprint1_dir = Path(args.sprint1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

<<<<<<<< HEAD:scripts/run_sprint2.py
    short = model_short_name(args.model)
    logger.info("=" * 70)
    logger.info(f"SPRINT 2 — {args.model}")
========
    # Resolve model ID (support shortcuts)
    if args.model in QWEN35_MODELS:
        model_id = QWEN35_MODELS[args.model]
        short_name = f"qwen3.5-{args.model}" if "3.5" in model_id else f"qwen2.5-{args.model}"
    else:
        model_id = args.model
        short_name = model_id.split("/")[-1].lower()

    logger.info("=" * 70)
    logger.info("SPRINT 2 — GEOMETRIC MEMORY (SCAR BUFFER)")
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    logger.info("=" * 70)
    logger.info(f"Sprint 1 dir : {sprint1_dir}")
    logger.info(f"Model short name: {short}")
    logger.info(f"Output : {output_dir}")
    logger.info(f"Capture experts: {args.capture_experts}")

    monitor = SystemMonitor(log_interval=60)

<<<<<<<< HEAD:scripts/run_sprint2.py
    # Locate Sprint 1 model directory
    sprint1_model_dir = sprint1_dir / short
    if not sprint1_model_dir.exists():
        # Fallback: try without hyphens? usually it's with hyphens
        alt = sprint1_dir / short.replace("-", "_")
        if alt.exists():
            sprint1_model_dir = alt
        else:
            raise FileNotFoundError(f"Could not find Sprint 1 results for {short} in {sprint1_dir}")

    logger.info(f"Sprint 1 model directory: {sprint1_model_dir}")

    # Load Â detector
========
    # Load Sprint 1 detector
    sprint1_model_dir = sprint1_dir / short_name
    if not sprint1_model_dir.exists():
        # Try fallback
        fallback = short_name.replace("qwen3.5-", "qwen3-").replace("qwen2.5-", "qwen3-")
        sprint1_model_dir = sprint1_dir / fallback
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    a_detector = AHatDetector.from_sprint0(sprint1_model_dir)

    # Load traces for training
    traces_path = sprint1_model_dir / "traces_sprint1.json"
    if not traces_path.exists():
        traces_path = sprint1_model_dir / "traces.json"
    if not traces_path.exists():
        raise FileNotFoundError(f"No traces found in {sprint1_model_dir}")

    # Determine hidden_dim
    with open(traces_path) as f:
        traces = json.load(f)
    hidden_dim = None
    for trace in traces:
        for step in trace.get("steps", []):
            hs_path = step.get("hidden_state_path")
            if hs_path and Path(hs_path).exists():
                h = np.load(hs_path)
                hidden_dim = h.shape[0]
                break
        if hidden_dim:
            break
    if hidden_dim is None:
        raise ValueError("Could not determine hidden_dim from traces")

    tool_names = get_tool_names()
    embed_dim = len(tool_names)

<<<<<<<< HEAD:scripts/run_sprint2.py
    # Predictor
    predictor_path = output_dir / short / "predictor.pt"
========
    # Number of experts (for MoE models)
    num_experts = 8 if "35b" in short_name or "122b" in short_name else None

    # Train or load predictor
    predictor_path = output_dir / short_name / "predictor.pt"
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    if args.skip_training and predictor_path.exists():
        predictor = ActionPredictor(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_experts=num_experts,
        )
        predictor.load(predictor_path)
        logger.info("Predictor loaded from disk")
    else:
<<<<<<<< HEAD:scripts/run_sprint2.py
        logger.info("Training MoE predictor...")
========
        logger.info("Training predictor on Sprint 1 traces...")
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
        predictor = train_predictor_from_traces(
            traces_path=traces_path,
            hidden_states_dir=sprint1_model_dir / "hidden_states",
            tool_names=tool_names,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_experts=num_experts,
            epochs=args.predictor_epochs,
            device=args.device,
        )
        predictor.save(predictor_path)

<<<<<<<< HEAD:scripts/run_sprint2.py
    # Scar buffer
========
    # Initialize scar buffer
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    scar_buffer = ScarBuffer(
        max_size=64,
        similarity_threshold=0.80,
        magnitude_threshold=0.1,
        decay_rate=0.95,
        kernel_bandwidth=0.1,
        tool_match_required=False,
    )

<<<<<<<< HEAD:scripts/run_sprint2.py
    # Integrator
    integrator = MoEGeometricIntegrator(
========
    # Initialize integrator
    integrator = GeometricIntegrator(
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
        predictor=predictor,
        scar_buffer=scar_buffer,
        scar_threshold=0.5,
    )

    # Select tasks
    if args.tasks == "single":
        task_list = SINGLE_TASKS
    elif args.tasks == "chain":
        task_list = CHAIN_TASKS
    elif args.tasks == "adversarial":
        task_list = ADVERSARIAL_TASKS
    else:
        task_list = ALL_TASKS

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
<<<<<<<< HEAD:scripts/run_sprint2.py

    logger.info(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
========
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else 32

    # Logger
    hs_logger = SavingHiddenStateLogger(
        model, num_layers,
        output_dir / short / "hs_sprint2"
    )

    # Agent
    agent = Sprint2Agent(
        model=model,
        tokenizer=tokenizer,
        tools=TOOL_MAP,
        a_detector=a_detector,
        integrator=integrator,
        hs_logger=hs_logger,
        max_steps=args.max_steps,
        temperature=args.temperature,
        device=args.device,
        capture_experts=args.capture_experts,
    )

    # Run
    traces = []
    for i, task in enumerate(task_list):
        logger.info(f"  [{i+1}/{len(task_list)}] {task.id}...")
        monitor.check(force=True)

        trace = agent.run(task.id, task.prompt, task.category, task.expected_tools)
        traces.append(trace)

        logger.info(f"    ✓ steps={len(trace.steps)} tools={trace.tools_used}")

        # Save intermediate results
        if (i + 1) % 10 == 0:
            out_path = output_dir / short / "traces_partial.json"
            with open(out_path, "w") as f:
                json.dump([t.to_dict() for t in traces], f, indent=2)

        force_cleanup()

    # Save final traces
<<<<<<<< HEAD:scripts/run_sprint2.py
    model_out = output_dir / short
========
    model_out = output_dir / short_name
>>>>>>>> 9ef3062e23f36f2811f23f9a66a6e0be49554a0f:run_sprint2.py
    model_out.mkdir(parents=True, exist_ok=True)
    with open(model_out / "traces_sprint2.json", "w") as f:
        json.dump([t.to_dict() for t in traces], f, indent=2)

    # Analyze
    analysis = analyze_traces(traces, task_list)
    logger.info(f"Success rate: {analysis['success_rate']:.1%}")

    # Save integrator stats
    integrator.save(model_out / "integrator")

    # Cleanup
    hs_logger.cleanup()
    del model, tokenizer, agent
    force_cleanup()

    logger.info(f"\nSprint 2 complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
