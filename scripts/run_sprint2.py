#!/usr/bin/env python3
"""
run_sprint2.py — Sprint 2: Geometric Memory (Scar Buffer)

Supports Qwen 3.5 models with MoE architecture.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.scar_buffer import ScarBuffer
from memory.predictor import ActionPredictor
from memory.integrator import GeometricIntegrator
from baselines.tools import TOOL_MAP, get_tool_names
from baselines.tasks import ALL_TASKS, SINGLE_TASKS, CHAIN_TASKS, ADVERSARIAL_TASKS
from baselines.react_agent import ReActAgent, HiddenStateLogger
from baselines.failure_analysis import analyze_traces

# Import from run_sprint1
from scripts.run_sprint1 import (
    Sprint1Agent, SavingHiddenStateLogger, SystemMonitor,
    force_cleanup, AHatDetector
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("sprint2.log")],
)
logger = logging.getLogger(__name__)


# ============================================================================
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
        capture_experts: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            tools=tools,
            a_detector=a_detector,
            hs_logger=hs_logger,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        self.integrator = integrator
        self.capture_experts = capture_experts
        self.last_expert_routing = None

    def _generate(self, messages):
        """Override to optionally capture expert routing."""
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

        # Generation kwargs
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.temperature > 0 else None,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        # Add expert routing capture if supported
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

        # Logits
        logits_last = None
        if hasattr(outputs, 'scores') and outputs.scores:
            logits_last = outputs.scores[-1][0]

        # Expert routing
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

            # 1. If agent generated a valid tool call
            if tool_name_text and tool_name_text in self.tools:
                actual_tool = tool_name_text
                actual_params = params_text
                routing_source = "textual"
                logger.info(f"📝 Textual tool call: {actual_tool}")

            # 2. If Â triggered and no final answer
            elif a_hat_triggered and not final_answer:
                actual_tool = "web_search"
                actual_params = {"query": "default search"}
                routing_source = "forced"
                logger.info(f"🔥 Â FORCED tool call: {actual_tool} (confidence={a_hat_confidence:.2f})")

            # 3. If final answer without forced tool
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

            # 4. Execute tool if we have one
            if actual_tool:
                # Tool embedding (one-hot)
                tool_names = get_tool_names()
                z_tool = np.zeros(len(tool_names))
                if actual_tool in tool_names:
                    z_tool[tool_names.index(actual_tool)] = 1.0

                # Predict next state with scar correction
                h_pred, corrections = self.integrator.predict(
                    h_before=h,
                    z_tool=z_tool,
                    tool_name=actual_tool,
                    expert_routing=expert_routing,
                )

                # Execute
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

            # Learn from previous step
            if step_idx > 0 and trace.steps[-2].action is not None:
                prev_step = trace.steps[-2]
                if prev_step.hidden_state_path and Path(prev_step.hidden_state_path).exists():
                    h_before = np.load(prev_step.hidden_state_path)

                    # Tool embedding for previous tool
                    tool_names = get_tool_names()
                    z_tool = np.zeros(len(tool_names))
                    if prev_step.action in tool_names:
                        z_tool[tool_names.index(prev_step.action)] = 1.0

                    # We don't have the prediction, but we can still learn from error
                    # This would need h_pred stored - for now, skip learning
                    pass

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
) -> ActionPredictor:
    """Train ActionPredictor on Sprint 1 traces."""
    import torch.optim as optim
    import torch.nn as nn

    with open(traces_path) as f:
        traces = json.load(f)

    # Collect (h_before, z_tool, h_after) triples
    X_h = []
    X_z = []
    Y = []

    for trace in traces:
        for i, step in enumerate(trace["steps"]):
            if step["action"] is None:
                continue
            if i + 1 >= len(trace["steps"]):
                continue

            # h_before
            hs_path = step.get("hidden_state_path")
            if not hs_path or not Path(hs_path).exists():
                continue
            h_before = np.load(hidden_states_dir / Path(hs_path).name)

            # h_after (next step)
            next_step = trace["steps"][i + 1]
            next_hs_path = next_step.get("hidden_state_path")
            if not next_hs_path or not Path(next_hs_path).exists():
                continue
            h_after = np.load(hidden_states_dir / Path(next_hs_path).name)

            # z_tool (one-hot)
            z_tool = np.zeros(len(tool_names))
            if step["action"] in tool_names:
                z_tool[tool_names.index(step["action"])] = 1.0

            X_h.append(h_before)
            X_z.append(z_tool)
            Y.append(h_after)

    if len(X_h) < 10:
        raise ValueError(f"Not enough training data: {len(X_h)} samples")

    logger.info(f"Training predictor on {len(X_h)} samples")

    # Convert to tensors
    X_h_t = torch.tensor(np.stack(X_h), dtype=torch.float32).to(device)
    X_z_t = torch.tensor(np.stack(X_z), dtype=torch.float32).to(device)
    Y_t = torch.tensor(np.stack(Y), dtype=torch.float32).to(device)

    # Initialize predictor
    predictor = ActionPredictor(
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_experts=num_experts,
    ).to(device)

    optimizer = optim.AdamW(predictor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    batch_size = 32

    for epoch in range(epochs):
        perm = torch.randperm(len(X_h))
        epoch_loss = 0.0

        for i in range(0, len(X_h), batch_size):
            idx = perm[i:i+batch_size]
            h_batch = X_h_t[idx]
            z_batch = X_z_t[idx]
            y_batch = Y_t[idx]

            optimizer.zero_grad()
            h_pred, _ = predictor(h_batch, z_batch)
            loss = criterion(h_pred, y_batch)
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
    parser.add_argument("--sprint1-dir", type=str, required=True,
                        help="Directory with Sprint 1 results")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model ID or size key (35b, 72b, 122b for Qwen 3.5)")
    parser.add_argument("--tasks", type=str, default="single",
                        choices=["single", "chain", "adversarial", "all"])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/sprint2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--predictor-epochs", type=int, default=50)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--capture-experts", action="store_true",
                        help="Capture expert routing (for Qwen 3.5 MoE models)")
    return parser.parse_args()


def main():
    args = parse_args()
    sprint1_dir = Path(args.sprint1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model ID (support shortcuts)
    if args.model in QWEN35_MODELS:
        model_id = QWEN35_MODELS[args.model]
        short_name = f"qwen3.5-{args.model}" if "3.5" in model_id else f"qwen2.5-{args.model}"
    else:
        model_id = args.model
        short_name = model_id.split("/")[-1].lower()

    logger.info("=" * 70)
    logger.info("SPRINT 2 — GEOMETRIC MEMORY (SCAR BUFFER)")
    logger.info("=" * 70)
    logger.info(f"Sprint 1 dir : {sprint1_dir}")
    logger.info(f"Model : {model_id}")
    logger.info(f"Output : {output_dir}")
    logger.info(f"Capture experts: {args.capture_experts}")

    monitor = SystemMonitor(log_interval=60)

    # Load Sprint 1 detector
    sprint1_model_dir = sprint1_dir / short_name
    if not sprint1_model_dir.exists():
        # Try fallback
        fallback = short_name.replace("qwen3.5-", "qwen3-").replace("qwen2.5-", "qwen3-")
        sprint1_model_dir = sprint1_dir / fallback
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

    # Number of experts (for MoE models)
    num_experts = 8 if "35b" in short_name or "122b" in short_name else None

    # Train or load predictor
    predictor_path = output_dir / short_name / "predictor.pt"
    if args.skip_training and predictor_path.exists():
        predictor = ActionPredictor(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_experts=num_experts,
        )
        predictor.load(predictor_path)
        logger.info("Predictor loaded from disk")
    else:
        logger.info("Training predictor on Sprint 1 traces...")
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

    # Initialize scar buffer
    scar_buffer = ScarBuffer(
        max_size=64,
        similarity_threshold=0.80,
        magnitude_threshold=0.1,
        decay_rate=0.95,
        kernel_bandwidth=0.1,
        tool_match_required=False,
    )

    # Initialize integrator
    integrator = GeometricIntegrator(
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
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
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
        output_dir / short_name / "hs_sprint2"
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
            out_path = output_dir / short_name / "traces_partial.json"
            with open(out_path, "w") as f:
                json.dump([t.to_dict() for t in traces], f, indent=2)

        force_cleanup()

    # Save final traces
    model_out = output_dir / short_name
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
