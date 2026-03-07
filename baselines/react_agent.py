"""
react_agent.py
Agent ReAct baseline avec logging des hidden states à chaque step.

Architecture :
  - Boucle ReAct classique : Thought → Action → Observation → repeat
  - Parsing du tool call depuis la génération textuelle
  - Hook sur le forward pass pour capturer les hidden states
  - Chaque step est loggé dans une trace structurée

Le logger de hidden states rend Sprint 0C (analyse Â) gratuit
une fois les traces de Sprint 0A générées.

Convention : docstrings français, code anglais.
"""

import json
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ─── Trace data structures ──────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Enregistrement d'un step de l'agent."""
    step_idx: int
    input_text: str  # Le prompt complet à ce step
    thought: str  # La réflexion générée
    action: Optional[str]  # Nom du tool (None si pas d'action)
    action_params: Optional[dict]  # Paramètres du tool call
    observation: Optional[str]  # Résultat du tool
    generated_text: str  # Texte brut généré
    # Hidden states (sauvegardés séparément en .npy)
    hidden_state_path: Optional[str] = None  # Chemin vers le .npy
    hidden_state_layer: Optional[int] = None  # Couche extraite
    # Timing
    generation_time_ms: float = 0.0
    tool_execution_time_ms: float = 0.0
    # Metadata pour l'analyse d'échecs (rempli pendant l'évaluation)
    failure_mode: Optional[str] = None  # wrong_tool, wrong_timing, loop, etc.


@dataclass
class Trace:
    """Trace complète d'une tâche."""
    task_id: str
    task_prompt: str
    task_category: str  # Catégorie de la tâche
    expected_tools: list[str]  # Tools attendus (ground truth)
    steps: list[StepRecord] = field(default_factory=list)
    success: bool = False
    final_answer: Optional[str] = None
    total_time_ms: float = 0.0
    num_tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Sérialise la trace (sans les hidden states, qui sont en .npy)."""
        return {
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "task_category": self.task_category,
            "expected_tools": self.expected_tools,
            "success": self.success,
            "final_answer": self.final_answer,
            "total_time_ms": self.total_time_ms,
            "num_tool_calls": self.num_tool_calls,
            "tools_used": self.tools_used,
            "error": self.error,
            "steps": [
                {
                    "step_idx": s.step_idx,
                    "thought": s.thought,
                    "action": s.action,
                    "action_params": s.action_params,
                    "observation": s.observation,
                    "generated_text": s.generated_text,
                    "hidden_state_path": s.hidden_state_path,
                    "hidden_state_layer": s.hidden_state_layer,
                    "generation_time_ms": s.generation_time_ms,
                    "tool_execution_time_ms": s.tool_execution_time_ms,
                    "failure_mode": s.failure_mode,
                }
                for s in self.steps
            ],
        }


# ─── Hidden State Logger ────────────────────────────────────────────────────

class HiddenStateLogger:
    """
    Capture les hidden states pendant le forward pass du LLM.
    
    Utilise un forward hook sur la couche médiane (depth 0.5).
    Chaque appel au modèle capture le hidden state mean-poolé.
    Les états sont sauvegardés en .npy pour l'analyse Sprint 0C.
    """

    def __init__(self, model, num_layers: int, output_dir: Path):
        self.model = model
        self.mid_layer = num_layers // 2
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._captured: Optional[torch.Tensor] = None
        self._hook = None
        self._setup_hook()

    def _setup_hook(self):
        """Installe le forward hook sur la couche médiane."""
        # Compatible avec la plupart des architectures HF
        layers = None
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "layers"):
                layers = inner.layers  # Llama, Mistral, Qwen
            elif hasattr(inner, "decoder"):
                if hasattr(inner.decoder, "layers"):
                    layers = inner.decoder.layers
        
        if layers is None:
            logger.warning("Cannot find model layers — hidden state logging disabled")
            return

        target_layer = layers[self.mid_layer]
        self._hook = target_layer.register_forward_hook(self._hook_fn)
        logger.info(f"Hidden state hook installed on layer {self.mid_layer}")

    def _hook_fn(self, module, input, output):
        """Callback du forward hook."""
        # output peut être un tuple (hidden_states, ...) ou un BaseModelOutput
        if isinstance(output, tuple):
            hidden = output[0]
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        else:
            hidden = output
        self._captured = hidden.detach()

    def get_last_hidden_state(self, attention_mask: Optional[torch.Tensor] = None) -> Optional[np.ndarray]:
        """
        Retourne le dernier hidden state capturé, mean-poolé.
        
        Args:
            attention_mask: masque d'attention pour le mean pooling
            
        Returns:
            vecteur numpy (hidden_dim,) ou None si pas de capture
        """
        if self._captured is None:
            return None
        
        h = self._captured  # (batch, seq_len, hidden_dim)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(h.device, h.dtype)
            h_masked = h * mask
            h_mean = h_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            h_mean = h.mean(dim=1)

        return h_mean.squeeze(0).cpu().float().numpy()

    def save(self, task_id: str, step_idx: int, h: np.ndarray) -> str:
        """Sauvegarde un hidden state et retourne le chemin."""
        filename = f"{task_id}_step{step_idx:03d}.npy"
        path = self.output_dir / filename
        np.save(path, h)
        return str(path)

    def cleanup(self):
        """Retire le hook."""
        if self._hook is not None:
            self._hook.remove()


# ─── ReAct Agent ─────────────────────────────────────────────────────────────

# System prompt pour le ReAct agent
REACT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools. 
To solve the user's task, use the ReAct framework:

1. **Thought**: Analyze what needs to be done next.
2. **Action**: Call a tool if needed, using the exact format below.
3. **Observation**: Read the tool's result.
4. Repeat until you can give a final answer.

## Tool Call Format
When you need to use a tool, write EXACTLY:
```
Thought: [your reasoning]
Action: tool_name(param1="value1", param2="value2")
```

When you have the final answer:
```
Thought: [your reasoning]
Final Answer: [your answer]
```

## Available Tools
{tool_descriptions}

## Rules
- Call ONE tool per step.
- Always think before acting.
- Use tools when you need external information or computation.
- Give a Final Answer when the task is complete.
- If a tool returns an error, try a different approach.
"""


class ReActAgent:
    """
    Agent ReAct baseline avec logging des hidden states.
    
    L'agent :
    1. Reçoit une tâche
    2. Génère une réflexion (Thought)
    3. Décide d'appeler un tool (Action) ou de répondre (Final Answer)
    4. Si tool call : exécute le tool, ajoute le résultat, continue
    5. Si final answer : termine
    
    À chaque step, le hidden state de la couche médiane est capturé et sauvegardé.
    """

    def __init__(
        self,
        model,
        tokenizer,
        tools: dict[str, Any],  # {name: Tool}
        hs_logger: HiddenStateLogger,
        max_steps: int = 10,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.hs_logger = hs_logger
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

        # Construire le system prompt avec la description des tools
        from baselines.tools import tool_description_block
        self.system_prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_description_block()
        )

    def _generate(self, messages: list[dict]) -> tuple[str, float]:
        """
        Génère une réponse à partir des messages.
        Retourne (texte_généré, temps_ms).
        """
        # Formatter les messages pour le modèle
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback pour les modèles sans chat template
            prompt = "\n".join(
                f"{'System' if m['role'] == 'system' else 'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in messages
            ) + "\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        t1 = time.perf_counter()

        # Décoder seulement les tokens générés (pas le prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Capturer le hidden state (le hook a été triggered par generate)
        h = self.hs_logger.get_last_hidden_state(inputs.get("attention_mask"))

        return text, (t1 - t0) * 1000, h, inputs.get("attention_mask")

    def _parse_action(self, text: str) -> tuple[Optional[str], Optional[dict]]:
        """
        Parse un tool call depuis le texte généré.
        Format attendu : tool_name(param1="value1", param2="value2")
        
        Retourne (tool_name, params_dict) ou (None, None).
        """
        # Pattern : Action: tool_name(...)
        action_match = re.search(
            r'Action:\s*(\w+)\(([^)]*)\)', text, re.IGNORECASE
        )
        if not action_match:
            # Essayer aussi le format sans "Action:"
            action_match = re.search(r'(\w+)\(([^)]*)\)', text)
            if not action_match:
                return None, None

        tool_name = action_match.group(1)
        params_str = action_match.group(2).strip()

        if tool_name not in self.tools:
            return tool_name, None  # Tool inconnu — sera loggé comme hallucination

        # Parser les paramètres
        params = {}
        if params_str:
            # Essayer le parsing par regex des kwargs
            for param_match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', params_str):
                params[param_match.group(1)] = param_match.group(2)
            # Aussi essayer les valeurs numériques
            for param_match in re.finditer(r'(\w+)\s*=\s*(\d+(?:\.\d+)?)', params_str):
                key = param_match.group(1)
                if key not in params:  # Ne pas écraser une valeur string
                    val = param_match.group(2)
                    params[key] = float(val) if "." in val else int(val)

        return tool_name, params

    def _parse_final_answer(self, text: str) -> Optional[str]:
        """Extrait la réponse finale si présente."""
        match = re.search(r'Final Answer:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _parse_thought(self, text: str) -> str:
        """Extrait la réflexion."""
        match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.split("\n")[0].strip()

    def run(self, task_id: str, task_prompt: str, task_category: str = "",
            expected_tools: list[str] = None) -> Trace:
        """
        Exécute l'agent sur une tâche. Retourne la trace complète.
        """
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

        for step_idx in range(self.max_steps):
            # Générer
            text, gen_time_ms, h, attn_mask = self._generate(messages)
            
            # Sauvegarder le hidden state
            hs_path = None
            if h is not None:
                hs_path = self.hs_logger.save(task_id, step_idx, h)

            # Parser
            thought = self._parse_thought(text)
            final_answer = self._parse_final_answer(text)

            if final_answer is not None:
                # Terminé
                step = StepRecord(
                    step_idx=step_idx,
                    input_text=messages[-1]["content"],
                    thought=thought,
                    action=None,
                    action_params=None,
                    observation=None,
                    generated_text=text,
                    hidden_state_path=hs_path,
                    hidden_state_layer=self.hs_logger.mid_layer,
                    generation_time_ms=gen_time_ms,
                )
                trace.steps.append(step)
                trace.final_answer = final_answer
                break

            # Parser le tool call
            tool_name, params = self._parse_action(text)

            if tool_name is None:
                # Pas de tool call ni de final answer — forcer la continuation
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "Please continue. Use a tool or provide a Final Answer."})

                step = StepRecord(
                    step_idx=step_idx,
                    input_text=messages[-3]["content"] if len(messages) > 2 else "",
                    thought=thought,
                    action=None,
                    action_params=None,
                    observation=None,
                    generated_text=text,
                    hidden_state_path=hs_path,
                    hidden_state_layer=self.hs_logger.mid_layer,
                    generation_time_ms=gen_time_ms,
                )
                trace.steps.append(step)
                continue

            # Exécuter le tool
            observation = ""
            exec_time = 0.0
            if tool_name in self.tools and params is not None:
                t_exec = time.perf_counter()
                observation = self.tools[tool_name].execute(**params)
                exec_time = (time.perf_counter() - t_exec) * 1000
                trace.num_tool_calls += 1
                trace.tools_used.append(tool_name)
            elif tool_name not in self.tools:
                observation = f"ERROR: Tool '{tool_name}' does not exist. Available tools: {', '.join(self.tools.keys())}"
            else:
                observation = f"ERROR: Could not parse parameters for tool '{tool_name}'."

            # Logger le step
            step = StepRecord(
                step_idx=step_idx,
                input_text=messages[-1]["content"],
                thought=thought,
                action=tool_name,
                action_params=params,
                observation=observation,
                generated_text=text,
                hidden_state_path=hs_path,
                hidden_state_layer=self.hs_logger.mid_layer,
                generation_time_ms=gen_time_ms,
                tool_execution_time_ms=exec_time,
            )
            trace.steps.append(step)

            # Ajouter le résultat au contexte
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        trace.total_time_ms = (time.perf_counter() - t_start) * 1000

        # Vérifier si la tâche n'a pas abouti
        if trace.final_answer is None:
            trace.error = "max_steps_reached"

        return trace
