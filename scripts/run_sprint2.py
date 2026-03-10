#!/usr/bin/env python3
"""
run_sprint2_qwen3.5.py — Sprint 2 pour Qwen 3.5 avec mémoire géométrique

Utilise les modèles Qwen 3.5 (35B-A3B et 122B-A10B) pour valider
l'approche du Scar Buffer sur architecture MoE.
"""

import argparse
import json
import logging
import sys
import time
import yaml
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.scar_buffer import ScarBuffer
from memory.predictor_moe import MoEAwarePredictor
from memory.integrator_moe import MoEGeometricIntegrator
from baselines.tools import TOOL_MAP, get_tool_names
from baselines.tasks import ALL_TASKS, SINGLE_TASKS, CHAIN_TASKS, ADVERSARIAL_TASKS
from baselines.failure_analysis import analyze_traces

# Import adapté de run_sprint1
from scripts.run_sprint1 import Sprint1Agent, SavingHiddenStateLogger, SystemMonitor, force_cleanup, AHatDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("sprint2_qwen3.5.log")],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Qwen 3.5
# ============================================================================

QWEN35_MODELS = {
    "35b": "Qwen/Qwen3.5-35B-A3B",      # 35B params, 3B activés
    "72b": "Qwen/Qwen3.5-72B",          # dense, pour comparaison
    "122b": "Qwen/Qwen3.5-122B-A10B",   # 122B params, 10B activés
}


# ============================================================================
# Agent Sprint 2 pour Qwen 3.5
# ============================================================================

class Sprint2Qwen35Agent(Sprint1Agent):
    """Agent avec Â gating + mémoire géométrique, optimisé pour Qwen 3.5."""

    def __init__(
        self,
        model,
        tokenizer,
        tools,
        a_detector,
        integrator: MoEGeometricIntegrator,
        hs_logger,
        max_steps: int = 10,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        device: str = "cuda",
        capture_experts: bool = True,   # Nouveau : capture les routages d'experts
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
        """Override pour capturer les routages d'experts."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{'System' if m['role']=='system' else 'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                               for m in messages) + "\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        t0 = time.perf_counter()

        # Pour capturer les routages d'experts, on utilise output_router_logits=True
        # (si supporté par le modèle)
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.temperature > 0 else None,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        # Ajouter output_router_logits si demandé
        if self.capture_experts:
            try:
                generate_kwargs["output_router_logits"] = True
            except:
                logger.warning("output_router_logits not supported, expert capture disabled")
                self.capture_experts = False

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        gen_time = (time.perf_counter() - t0) * 1000

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0, input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Logits du dernier token
        logits_last = None
        if hasattr(outputs, 'scores') and outputs.scores:
            logits_last = outputs.scores[-1][0]

        # Routage d'experts (si disponible)
        expert_routing = None
        if self.capture_experts and hasattr(outputs, 'router_logits') and outputs.router_logits:
            # Prendre les logits du dernier token pour la dernière couche
            expert_routing = outputs.router_logits[-1][0].cpu().numpy()
            self.last_expert_routing = expert_routing

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

        # Pour stocker les routages d'experts dans les steps
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

            # ─── Décision : quel outil utiliser ? ───
            actual_tool = None
            actual_params = None
            routing_source = "none"

            # 1. Si l'agent a généré un tool call valide
            if tool_name_text and tool_name_text in self.tools:
                actual_tool = tool_name_text
                actual_params = params_text
                routing_source = "textual"
                logger.info(f"📝 Textual tool call: {actual_tool}")

            # 2. Si Â déclenché et pas de réponse finale
            elif a_hat_triggered and not final_answer:
                actual_tool = "web_search"
                actual_params = {"query": "default search"}
                routing_source = "forced"
                logger.info(f"🔥 Â FORCED tool call: {actual_tool} (confidence={a_hat_confidence:.2f})")

            # 3. Si réponse finale sans tool forcé
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

            # 4. Exécuter l'outil si on en a un
            if actual_tool:
                # Embedding de l'outil (one-hot)
                tool_names = get_tool_names()
                z_tool = np.zeros(len(tool_names))
                if actual_tool in tool_names:
                    z_tool[tool_names.index(actual_tool)] = 1.0

                # Prédiction du prochain état (avec correction)
                h_pred, corrections, expert_pred = self.integrator.predict(
                    h_before=h,
                    z_tool=z_tool,
                    tool_name=actual_tool,
                    expert_routing_before=expert_routing,
                )

                # Exécution
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

                # Ajouter au contexte
                if routing_source != "textual":
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content":
                        f"[Geometric routing selected tool '{actual_tool}']\n"
                        f"Observation: {observation}"})
                else:
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})

            else:
                # Pas de tool, pas de réponse finale → forcer continuation
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

            # Apprentissage à partir du step précédent
            if step_idx > 0 and trace.steps[-2].action is not None:
                prev_step = trace.steps[-2]
                if prev_step.hidden_state_path and Path(prev_step.hidden_state_path).exists():
                    h_before = np.load(prev_step.hidden_state_path)

                    # Embedding de l'outil précédent
                    tool_names = get_tool_names()
                    z_tool = np.zeros(len(tool_names))
                    if prev_step.action in tool_names:
                        z_tool[tool_names.index(prev_step.action)] = 1.0

                    # Routage d'experts avant/après
                    exp_before = expert_routings[step_idx-1] if step_idx-1 < len(expert_routings) else None
                    exp_after = expert_routing

                    # Apprentissage
                    self.integrator.learn(
                        h_before=h_before,
                        z_tool=z_tool,
                        tool_name=prev_step.action,
                        h_predicted=None,  # On n'a pas la prédiction stockée
                        h_actual=h,
                        expert_routing_before=exp_before,
                        expert_routing_after=exp_after,
                        task_id=task_id,
                    )

            self.integrator.step()

        trace.total_time_ms = (time.perf_counter() - t_start) * 1000
        if trace.final_answer is None:
            trace.error = "max_steps_reached"

        return trace


# ============================================================================
# Entraînement du prédicteur (adapté pour MoE)
# ============================================================================

def train_predictor_for_qwen35(
    traces_path: Path,
    hidden_states_dir: Path,
    tool_names: list[str],
    hidden_dim: int,
    num_experts: int = 8,
    embed_dim: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
) -> MoEAwarePredictor:
    """Entraîne le prédicteur sur les traces de Sprint 1 (Qwen 3.5)."""
    import torch.optim as optim
    import torch.nn as nn

    with open(traces_path) as f:
        traces = json.load(f)

    # Collecter (h_before, z_tool, h_after, expert_before, expert_after)
    X_h = []
    X_z = []
    Y_h = []
    Y_exp_before = []
    Y_exp_after = []

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

            # h_after
            next_step = trace["steps"][i + 1]
            next_hs_path = next_step.get("hidden_state_path")
            if not next_hs_path or not Path(next_hs_path).exists():
                continue
            h_after = np.load(hidden_states_dir / Path(next_hs_path).name)

            # z_tool (one-hot)
            z_tool = np.zeros(len(tool_names))
            if step["action"] in tool_names:
                z_tool[tool_names.index(step["action"])] = 1.0

            # Routages d'experts (simulés si non disponibles)
            # Dans les traces, on n'a pas cette info, on utilisera des vecteurs aléatoires
            # pour l'entraînement initial
            exp_before = np.random.dirichlet(np.ones(num_experts))
            exp_after = np.random.dirichlet(np.ones(num_experts))

            X_h.append(h_before)
            X_z.append(z_tool)
            Y_h.append(h_after)
            Y_exp_before.append(exp_before)
            Y_exp_after.append(exp_after)

    if len(X_h) < 10:
        raise ValueError(f"Pas assez de données: {len(X_h)} échantillons")

    logger.info(f"Entraînement du prédicteur MoE sur {len(X_h)} échantillons")

    # Conversion en tenseurs
    X_h_t = torch.tensor(np.stack(X_h), dtype=torch.float32).to(device)
    X_z_t = torch.tensor(np.stack(X_z), dtype=torch.float32).to(device)
    Y_h_t = torch.tensor(np.stack(Y_h), dtype=torch.float32).to(device)
    Y_exp_t = torch.tensor(np.stack(Y_exp_after), dtype=torch.float32).to(device)

    # Initialisation du prédicteur
    predictor = MoEAwarePredictor(
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_experts=num_experts,
    ).to(device)

    optimizer = optim.AdamW(predictor.parameters(), lr=lr)
    criterion_h = nn.MSELoss()
    criterion_exp = nn.KLDivLoss(reduction="batchmean")

    batch_size = 32
    n_batches = (len(X_h) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        perm = torch.randperm(len(X_h))
        epoch_loss_h = 0.0
        epoch_loss_exp = 0.0

        for i in range(0, len(X_h), batch_size):
            idx = perm[i:i+batch_size]
            h_batch = X_h_t[idx]
            z_batch = X_z_t[idx]
            y_batch = Y_h_t[idx]
            y_exp_batch = Y_exp_t[idx]
            exp_batch = torch.stack([Y_exp_before[j] for j in idx]).to(device)

            optimizer.zero_grad()
            h_pred, exp_pred = predictor(h_batch, z_batch, exp_batch)
            loss_h = criterion_h(h_pred, y_batch)
            loss_exp = criterion_exp(
                torch.log_softmax(exp_pred, dim=-1),
                torch.softmax(y_exp_batch, dim=-1)
            )
            loss = loss_h + 0.1 * loss_exp  # Pondération
            loss.backward()
            optimizer.step()

            epoch_loss_h += loss_h.item() * len(idx)
            epoch_loss_exp += loss_exp.item() * len(idx)

        epoch_loss_h /= len(X_h)
        epoch_loss_exp /= len(X_h)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: loss_h={epoch_loss_h:.6f}, loss_exp={epoch_loss_exp:.6f}")

    return predictor


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Sprint 2 pour Qwen 3.5")
    parser.add_argument("--sprint1-dir", type=str, required=True)
    parser.add_argument("--model-size", type=str, default="35b",
                        choices=["35b", "72b", "122b"])
    parser.add_argument("--tasks", type=str, default="single",
                        choices=["single", "chain", "adversarial", "all"])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/sprint2_qwen35")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--predictor-epochs", type=int, default=50)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--num-experts", type=int, default=8,
                        help="Nombre d'experts dans le MoE")
    return parser.parse_args()


def main():
    args = parse_args()
    sprint1_dir = Path(args.sprint1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = QWEN35_MODELS[args.model_size]
    short_name = f"qwen3.5-{args.model_size}"

    logger.info("=" * 70)
    logger.info(f"SPRINT 2 — Qwen 3.5 ({args.model_size})")
    logger.info("=" * 70)
    logger.info(f"Sprint 1 dir : {sprint1_dir}")
    logger.info(f"Model : {model_id}")
    logger.info(f"Output : {output_dir}")

    monitor = SystemMonitor(log_interval=60)

    # Charger le détecteur Â (depuis Sprint 1)
    from scripts.run_sprint1 import AHatDetector
    sprint1_model_dir = sprint1_dir / short_name.replace("-", "_")  # Adaptation
    if not sprint1_model_dir.exists():
        sprint1_model_dir = sprint1_dir / "qwen3-8b"  # Fallback
    a_detector = AHatDetector.from_sprint0(sprint1_model_dir)

    # Charger les traces pour l'entraînement
    traces_path = sprint1_model_dir / "traces_sprint1.json"
    if not traces_path.exists():
        traces_path = sprint1_model_dir / "traces.json"
    if not traces_path.exists():
        raise FileNotFoundError(f"Traces introuvables dans {sprint1_model_dir}")

    # Déterminer hidden_dim
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
        raise ValueError("Impossible de déterminer hidden_dim")

    tool_names = get_tool_names()
    embed_dim = len(tool_names)

    # Entraîner ou charger le prédicteur
    predictor_path = output_dir / short_name / "predictor.pt"
    if args.skip_training and predictor_path.exists():
        predictor = MoEAwarePredictor(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_experts=args.num_experts,
        )
        predictor.load(predictor_path)
        logger.info("Prédicteur chargé depuis le disque")
    else:
        logger.info("Entraînement du prédicteur MoE...")
        predictor = train_predictor_for_qwen35(
            traces_path=traces_path,
            hidden_states_dir=sprint1_model_dir / "hidden_states",
            tool_names=tool_names,
            hidden_dim=hidden_dim,
            num_experts=args.num_experts,
            embed_dim=embed_dim,
            epochs=args.predictor_epochs,
            device=args.device,
        )
        predictor.save(predictor_path)

    # Initialiser le Scar Buffer
    scar_buffer = ScarBuffer(
        max_size=64,
        similarity_threshold=0.80,
        magnitude_threshold=0.1,
        decay_rate=0.95,
        kernel_bandwidth=0.1,
        tool_match_required=False,
    )

    # Initialiser l'intégrateur MoE
    integrator = MoEGeometricIntegrator(
        predictor=predictor,
        scar_buffer=scar_buffer,
        scar_threshold=0.5,
        num_experts=args.num_experts,
    )

    # Sélectionner les tâches
    if args.tasks == "single":
        task_list = SINGLE_TASKS
    elif args.tasks == "chain":
        task_list = CHAIN_TASKS
    elif args.tasks == "adversarial":
        task_list = ADVERSARIAL_TASKS
    else:
        task_list = ALL_TASKS

    # Charger le modèle Qwen 3.5
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Chargement de {model_id}...")
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
    agent = Sprint2Qwen35Agent(
        model=model,
        tokenizer=tokenizer,
        tools=TOOL_MAP,
        a_detector=a_detector,
        integrator=integrator,
        hs_logger=hs_logger,
        max_steps=args.max_steps,
        temperature=args.temperature,
        device=args.device,
        capture_experts=True,
    )

    # Exécution
    traces = []
    for i, task in enumerate(task_list):
        logger.info(f"  [{i+1}/{len(task_list)}] {task.id}...")
        monitor.check(force=True)

        trace = agent.run(task.id, task.prompt, task.category, task.expected_tools)
        traces.append(trace)

        logger.info(f"    ✓ steps={len(trace.steps)} tools={trace.tools_used}")

        if (i + 1) % 10 == 0:
            out_path = output_dir / short_name / "traces_partial.json"
            with open(out_path, "w") as f:
                json.dump([t.to_dict() for t in traces], f, indent=2)

        force_cleanup()

    # Sauvegarde finale
    model_out = output_dir / short_name
    model_out.mkdir(parents=True, exist_ok=True)
    with open(model_out / "traces_sprint2.json", "w") as f:
        json.dump([t.to_dict() for t in traces], f, indent=2)

    # Analyse
    analysis = analyze_traces(traces, task_list)
    logger.info(f"Taux de succès: {analysis['success_rate']:.1%}")

    # Stats de l'intégrateur
    integrator.save(model_out / "integrator")
    with open(model_out / "integrator_stats.json", "w") as f:
        json.dump(integrator.get_stats(), f, indent=2, default=str)

    # Cleanup
    hs_logger.cleanup()
    del model, tokenizer, agent
    force_cleanup()

    logger.info(f"\nSprint 2 terminé. Résultats dans {output_dir}/")


if __name__ == "__main__":
    main()
