"""
run_sprint1.py
Sprint 1 — Â detector + routing géométrique.

Basé sur les résultats Sprint 0 :
  - AUC Â > 0.94 sur les 3 tailles → le signal est exploitable
  - no_tool est le mode d'échec dominant → Â detector en priorité
  - Les tools ne clustent PAS dans les hidden states médians → routing par logits ou contrastif
  
Sprint 1 construit deux composants :
  1. Â detector : projection linéaire sur la direction mean-diff → gating tool/no-tool
  2. Routing géométrique : sélection du tool par les logits top-k ou embedding contrastif

Usage (sur RunPod, après Sprint 0) :
  # Sprint 1 complet : entraîner Â + routing, benchmarker A/B
  python run_sprint1.py --sprint0-dir results/sprint0 --model Qwen/Qwen3-8B

  # Sprint 1 scaling : comparer le gain sur 1.7B vs 4B vs 8B
  python run_sprint1.py --sprint0-dir results/sprint0 --scaling-preset

  # Test rapide
  python run_sprint1.py --sprint0-dir results/sprint0 --model Qwen/Qwen3-8B --tasks single --max-steps 5

Convention : docstrings français, code anglais.
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sprint1.log"),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Â Detector — projection linéaire sur la direction mean-diff
# ═══════════════════════════════════════════════════════════════════════════

class AHatDetector:
    """
    Détecteur d'agentivité basé sur la direction mean-diff des hidden states.
    
    Mécanisme : projette h sur la direction Â (h_tool_mean - h_notool_mean)
    et compare au seuil θ. Si proj > θ → tool call recommandé.
    
    Calibré sur les traces Sprint 0.
    """

    def __init__(self, a_hat_direction: np.ndarray, threshold: float = 0.0):
        self.direction = a_hat_direction / (np.linalg.norm(a_hat_direction) + 1e-12)
        self.threshold = threshold
        self.direction_torch = None  # lazy init on first use

    @classmethod
    def from_sprint0(cls, sprint0_model_dir: Path):
        """Charge Â et calibre θ depuis les résultats Sprint 0."""
        # Charger la direction Â
        a_hat_path = sprint0_model_dir / "a_hat_extracted.npy"
        if not a_hat_path.exists():
            raise FileNotFoundError(f"Â direction not found: {a_hat_path}")
        a_hat = np.load(a_hat_path)
        logger.info(f"Â direction loaded: dim={a_hat.shape[0]}")

        # Charger les projections pour calibrer θ
        results_path = sprint0_model_dir / "a_hat_traces.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            # θ optimal = point milieu entre les moyennes tool et no-tool
            proj_tool = results.get("proj_tool_mean", 0)
            proj_notool = results.get("proj_notool_mean", 0)
            # Si les clés n'existent pas, calculer depuis separation
            if proj_tool and proj_notool:
                threshold = (proj_tool + proj_notool) / 2
            else:
                threshold = 0.0
            logger.info(f"Threshold calibrated: θ={threshold:.4f} "
                        f"(tool_mean={proj_tool:.4f}, notool_mean={proj_notool:.4f})")
        else:
            threshold = 0.0
            logger.warning("No calibration data — using θ=0")

        return cls(a_hat, threshold)

    def predict(self, h: np.ndarray) -> tuple[bool, float]:
        """
        Prédit si un tool call est recommandé.
        
        Returns:
            (should_call_tool, confidence)
            confidence > 0 = tool recommandé, < 0 = pas de tool
        """
        proj = float(np.dot(h, self.direction))
        return proj > self.threshold, proj - self.threshold

    def predict_torch(self, h: torch.Tensor) -> tuple[bool, float]:
        """Version torch pour usage dans la boucle de génération."""
        if self.direction_torch is None or self.direction_torch.device != h.device:
            self.direction_torch = torch.from_numpy(self.direction).to(h.device, h.dtype)
        proj = float(torch.dot(h.flatten(), self.direction_torch))
        return proj > self.threshold, proj - self.threshold


# ═══════════════════════════════════════════════════════════════════════════
# Logits-based Tool Router — matrice de confusion des logits
# ═══════════════════════════════════════════════════════════════════════════

class LogitsToolRouter:
    """
    Routing de tools basé sur les logits du modèle au moment du choix.
    
    Au lieu de laisser le décodeur textuel choisir le tool par génération,
    on regarde les logits au token de décision et on prend le tool avec
    la plus haute probabilité parmi les tools connus.
    
    Avantage : exploite directement la distribution du modèle sans
    dépendre du sampling textuel. Particulièrement utile quand le modèle
    "sait" le bon tool (il est dans les top-k logits) mais ne le génère
    pas (le sampling tombe sur un autre token).
    """

    def __init__(self, tokenizer, tool_names: list[str]):
        self.tokenizer = tokenizer
        self.tool_names = tool_names
        
        # Pré-calculer les token IDs de chaque tool
        self.tool_token_ids = {}
        self.tool_first_tokens = {}
        for name in tool_names:
            ids = tokenizer.encode(name, add_special_tokens=False)
            self.tool_token_ids[name] = ids
            if ids:
                self.tool_first_tokens[ids[0]] = name
        
        logger.info(f"LogitsToolRouter initialized with {len(tool_names)} tools, "
                     f"{len(self.tool_first_tokens)} unique first tokens")

    def route_from_logits(self, logits: torch.Tensor, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Étant donné les logits au dernier token, retourne les tools candidats
        ordonnés par probabilité.
        
        Args:
            logits: (vocab_size,) logits au dernier token
            top_k: nombre de candidats à retourner
            
        Returns:
            Liste de (tool_name, probability) ordonnée par prob décroissante
        """
        probs = torch.softmax(logits, dim=-1)
        
        # Scorer chaque tool
        tool_scores = []
        for name in self.tool_names:
            token_ids = self.tool_token_ids.get(name, [])
            if not token_ids:
                continue
            # Probabilité du premier token du nom du tool
            first_token_prob = float(probs[token_ids[0]])
            tool_scores.append((name, first_token_prob))
        
        # Trier par probabilité décroissante
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return tool_scores[:top_k]

    def get_entropy(self, logits: torch.Tensor) -> float:
        """Entropie des logits (mesure de confiance)."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return float(-(probs * log_probs).sum())


# ═══════════════════════════════════════════════════════════════════════════
# Contrastive Tool Embedding — apprend l'espace fonctionnel
# ═══════════════════════════════════════════════════════════════════════════

class ContrastiveToolEmbedding:
    """
    Embedding des tools appris par objectif contrastif sur les traces Sprint 0.
    
    Pour chaque step avec tool call dans les traces :
      - h_before est le hidden state avant le call
      - tool_correct est le tool effectivement appelé
      - tool_random est un tool aléatoire (négatif)
    
    On entraîne une projection linéaire h → z telle que
    cos(z, e_correct) > cos(z, e_random) (InfoNCE).
    """

    def __init__(self, hidden_dim: int, embed_dim: int = 128, n_tools: int = 25):
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_tools = n_tools
        
        # Projection linéaire h → z
        self.projector = None  # torch.nn.Linear, init au train
        # Embeddings des tools (appris)
        self.tool_embeddings = None  # torch.nn.Embedding
        self.tool_name_to_idx = {}
        self.idx_to_tool_name = {}
        self.trained = False

    def train_from_traces(
        self,
        traces: list[dict],
        hidden_states_dir: Path,
        tool_names: list[str],
        epochs: int = 50,
        lr: float = 1e-3,
        temperature: float = 0.1,
        device: str = "cuda",
    ) -> dict:
        """Entraîne l'embedding contrastif sur les traces Sprint 0."""
        import torch.nn as nn
        import torch.nn.functional as F

        # Mapping tool name → index
        self.tool_name_to_idx = {name: i for i, name in enumerate(tool_names)}
        self.idx_to_tool_name = {i: name for i, name in enumerate(tool_names)}
        self.n_tools = len(tool_names)

        # Collecter les paires (h_before, tool_idx) depuis les traces
        pairs = []
        for trace in traces:
            for step in trace.get("steps", []):
                if step.get("action") and step["action"] in self.tool_name_to_idx:
                    hs_path = step.get("hidden_state_path")
                    if hs_path and Path(hs_path).exists():
                        h = np.load(hs_path)
                        tool_idx = self.tool_name_to_idx[step["action"]]
                        pairs.append((h, tool_idx))

        if len(pairs) < 10:
            logger.warning(f"Only {len(pairs)} training pairs — insufficient for contrastive training")
            return {"error": "insufficient_data", "n_pairs": len(pairs)}

        logger.info(f"Training contrastive embedding on {len(pairs)} pairs, "
                     f"dim {self.hidden_dim} → {self.embed_dim}")

        # Préparer les données
        H = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32, device=device)
        tool_indices = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)

        # Modèle
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        ).to(device)

        self.tool_embeddings = nn.Embedding(self.n_tools, self.embed_dim).to(device)
        nn.init.normal_(self.tool_embeddings.weight, std=0.02)

        # Optimizer
        optimizer = torch.optim.AdamW(
            list(self.projector.parameters()) + list(self.tool_embeddings.parameters()),
            lr=lr,
        )

        # Training loop — InfoNCE
        best_loss = float("inf")
        losses = []

        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(len(H))
            H_shuffled = H[perm]
            tools_shuffled = tool_indices[perm]

            # Forward
            z = self.projector(H_shuffled)  # (N, embed_dim)
            z = F.normalize(z, dim=-1)

            e_all = F.normalize(self.tool_embeddings.weight, dim=-1)  # (n_tools, embed_dim)

            # Similarité avec tous les tools
            sim = z @ e_all.T / temperature  # (N, n_tools)

            # InfoNCE loss
            loss = F.cross_entropy(sim, tools_shuffled)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = float(loss)
            losses.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val

            if (epoch + 1) % 10 == 0:
                # Accuracy
                preds = sim.argmax(dim=-1)
                acc = float((preds == tools_shuffled).float().mean())
                logger.info(f"  Epoch {epoch+1}/{epochs}: loss={loss_val:.4f}, acc={acc:.3f}")

        # Évaluation finale
        with torch.no_grad():
            z = self.projector(H)
            z = F.normalize(z, dim=-1)
            e_all = F.normalize(self.tool_embeddings.weight, dim=-1)
            sim = z @ e_all.T
            preds = sim.argmax(dim=-1)
            acc_final = float((preds == tool_indices).float().mean())

            # Recall@3
            top3 = sim.topk(3, dim=-1).indices
            recall3 = float((top3 == tool_indices.unsqueeze(1)).any(dim=1).float().mean())

        self.trained = True
        logger.info(f"Training done: acc={acc_final:.3f}, recall@3={recall3:.3f}")

        return {
            "n_pairs": len(pairs),
            "final_accuracy": acc_final,
            "recall_at_3": recall3,
            "best_loss": best_loss,
            "losses": losses,
        }

    def route(self, h: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Route un hidden state vers les top-k tools."""
        if not self.trained:
            return []

        import torch.nn.functional as F

        with torch.no_grad():
            h_tensor = torch.tensor(h, dtype=torch.float32, device=next(self.projector.parameters()).device)
            z = self.projector(h_tensor.unsqueeze(0))
            z = F.normalize(z, dim=-1)
            e_all = F.normalize(self.tool_embeddings.weight, dim=-1)
            sim = (z @ e_all.T).squeeze(0)

            topk = sim.topk(top_k)
            results = [
                (self.idx_to_tool_name[int(idx)], float(score))
                for idx, score in zip(topk.indices, topk.values)
            ]
        return results

    def save(self, path: Path):
        """Sauvegarde l'embedding entraîné."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.projector.state_dict(), path / "projector.pt")
        torch.save(self.tool_embeddings.state_dict(), path / "tool_embeddings.pt")
        with open(path / "config.json", "w") as f:
            json.dump({
                "hidden_dim": self.hidden_dim,
                "embed_dim": self.embed_dim,
                "n_tools": self.n_tools,
                "tool_name_to_idx": self.tool_name_to_idx,
            }, f, indent=2)

    def load(self, path: Path, device: str = "cuda"):
        """Charge un embedding entraîné."""
        import torch.nn as nn
        with open(path / "config.json") as f:
            config = json.load(f)
        self.hidden_dim = config["hidden_dim"]
        self.embed_dim = config["embed_dim"]
        self.n_tools = config["n_tools"]
        self.tool_name_to_idx = config["tool_name_to_idx"]
        self.idx_to_tool_name = {int(v): k for k, v in self.tool_name_to_idx.items()}

        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        ).to(device)
        self.projector.load_state_dict(torch.load(path / "projector.pt", map_location=device))

        self.tool_embeddings = nn.Embedding(self.n_tools, self.embed_dim).to(device)
        self.tool_embeddings.load_state_dict(torch.load(path / "tool_embeddings.pt", map_location=device))
        self.trained = True


# ═══════════════════════════════════════════════════════════════════════════
# Agent augmenté Sprint 1 — Â gating + routing géométrique
# ═══════════════════════════════════════════════════════════════════════════

class Sprint1Agent:
    """
    Agent ReAct augmenté avec Â detector et routing géométrique.
    
    Deux modifications par rapport au baseline Sprint 0 :
    1. Â gating : si le hidden state projète au-dessus du seuil Â,
       on force un tool call même si le décodeur textuel n'en génère pas
    2. Routing géométrique : quand un tool call est forcé, le tool est
       choisi par le routeur (logits ou contrastif) plutôt que par
       le décodeur textuel
    """

    def __init__(
        self,
        model,
        tokenizer,
        tools: dict,
        a_detector: AHatDetector,
        tool_router: ContrastiveToolEmbedding,
        logits_router: LogitsToolRouter,
        hs_logger,
        max_steps: int = 10,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        device: str = "cuda",
        routing_mode: str = "contrastive",  # "contrastive", "logits", or "baseline"
        a_hat_enabled: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.a_detector = a_detector
        self.tool_router = tool_router
        self.logits_router = logits_router
        self.hs_logger = hs_logger
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.routing_mode = routing_mode
        self.a_hat_enabled = a_hat_enabled

        from baselines.tools import tool_description_block
        from baselines.react_agent import REACT_SYSTEM_PROMPT
        self.system_prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_description_block()
        )

    def run(self, task_id: str, task_prompt: str, task_category: str = "",
            expected_tools: list = None):
        """Exécute l'agent sur une tâche avec les augmentations Sprint 1."""
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

        for step_idx in range(self.max_steps):
            text, gen_time_ms, h, logits_last = self._generate(messages)

            # Sauvegarder le hidden state
            hs_path = None
            if h is not None:
                hs_path = self.hs_logger.save(task_id, step_idx, h)

            # ── Â detector ──
            a_hat_triggered = False
            a_hat_confidence = 0.0
            if self.a_hat_enabled and h is not None:
                should_call, confidence = self.a_detector.predict(h)
                a_hat_confidence = confidence
                if should_call:
                    a_hat_triggered = True

            # Parser la sortie textuelle
            thought = self._parse_thought(text)
            final_answer = self._parse_final_answer(text)
            tool_name_text, params_text = self._parse_action(text)

            # ── Décision de routing ──
            actual_tool = None
            actual_params = None
            routing_source = "none"

            if tool_name_text and tool_name_text in self.tools:
                # Le modèle a généré un tool call valide → l'utiliser
                actual_tool = tool_name_text
                actual_params = params_text
                routing_source = "textual"

            elif a_hat_triggered and not final_answer:
                # Â dit "tool call" mais le modèle n'en a pas généré → routing géométrique
                if self.routing_mode == "contrastive" and self.tool_router.trained:
                    candidates = self.tool_router.route(h, top_k=3)
                    if candidates:
                        actual_tool = candidates[0][0]
                        actual_params = {}
                        routing_source = "contrastive"
                elif self.routing_mode == "logits" and logits_last is not None:
                    candidates = self.logits_router.route_from_logits(logits_last, top_k=3)
                    if candidates and candidates[0][1] > 0.01:
                        actual_tool = candidates[0][0]
                        actual_params = {}
                        routing_source = "logits"

            # ── Final answer ──
            if final_answer and not a_hat_triggered:
                step = StepRecord(
                    step_idx=step_idx,
                    input_text=messages[-1]["content"],
                    thought=thought,
                    action=None,
                    action_params=None,
                    observation=None,
                    generated_text=text,
                    hidden_state_path=hs_path,
                    hidden_state_layer=self.hs_logger.mid_layer if hasattr(self.hs_logger, 'mid_layer') else None,
                    generation_time_ms=gen_time_ms,
                )
                trace.steps.append(step)
                trace.final_answer = final_answer
                break

            # ── Exécuter le tool ──
            observation = ""
            exec_time = 0.0

            if actual_tool and actual_tool in self.tools:
                t_exec = time.perf_counter()
                try:
                    observation = self.tools[actual_tool].execute(**(actual_params or {}))
                except Exception as e:
                    observation = f"ERROR: {e}"
                exec_time = (time.perf_counter() - t_exec) * 1000
                trace.num_tool_calls += 1
                trace.tools_used.append(actual_tool)
            elif actual_tool:
                observation = f"ERROR: Tool '{actual_tool}' not found."

            step = StepRecord(
                step_idx=step_idx,
                input_text=messages[-1]["content"],
                thought=thought,
                action=actual_tool,
                action_params=actual_params,
                observation=observation,
                generated_text=text,
                hidden_state_path=hs_path,
                hidden_state_layer=self.hs_logger.mid_layer if hasattr(self.hs_logger, 'mid_layer') else None,
                generation_time_ms=gen_time_ms,
                tool_execution_time_ms=exec_time,
            )
            trace.steps.append(step)

            # Ajouter au contexte
            if actual_tool and observation:
                if routing_source != "textual":
                    # Le tool a été choisi par le routing, pas par le texte
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content":
                        f"[Geometric routing selected tool '{actual_tool}']\n"
                        f"Observation: {observation}"})
                else:
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
            elif not actual_tool and not final_answer:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content":
                    "Please continue. Use a tool or provide a Final Answer."})

        trace.total_time_ms = (time.perf_counter() - t_start) * 1000
        if trace.final_answer is None:
            trace.error = "max_steps_reached"

        return trace

    def _generate(self, messages):
        """Génère avec capture des hidden states et logits."""
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
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=False,
                return_dict_in_generate=False,
            )
        gen_time = (time.perf_counter() - t0) * 1000

        input_len = inputs["input_ids"].shape[1]
        text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

        # Hidden state du hook
        h = self.hs_logger.get_last_hidden_state(inputs.get("attention_mask"))

        # Logits (forward pass additionnel sur le dernier token si nécessaire)
        logits_last = None
        # Note: pour le routing logits, on pourrait faire un forward pass séparé
        # mais c'est coûteux. Pour l'instant, on n'utilise les logits que si
        # routing_mode == "logits"

        return text, gen_time, h, logits_last

    def _parse_thought(self, text):
        import re
        m = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', text, re.I | re.S)
        return m.group(1).strip() if m else text.split("\n")[0].strip()

    def _parse_final_answer(self, text):
        import re
        m = re.search(r'Final Answer:\s*(.+)', text, re.I | re.S)
        return m.group(1).strip() if m else None

    def _parse_action(self, text):
        import re
        m = re.search(r'Action:\s*(\w+)\(([^)]*)\)', text, re.I)
        if not m:
            return None, None
        tool_name = m.group(1)
        params_str = m.group(2).strip()
        params = {}
        if params_str:
            for pm in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', params_str):
                params[pm.group(1)] = pm.group(2)
            for pm in re.finditer(r'(\w+)\s*=\s*(\d+(?:\.\d+)?)', params_str):
                if pm.group(1) not in params:
                    val = pm.group(2)
                    params[pm.group(1)] = float(val) if "." in val else int(val)
        return tool_name, params


# ═══════════════════════════════════════════════════════════════════════════
# Main — Sprint 1 runner
# ═══════════════════════════════════════════════════════════════════════════

SCALING_MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sprint 1 — Â detector + routing géométrique")
    parser.add_argument("--sprint0-dir", type=str, required=True,
                        help="Répertoire des résultats Sprint 0")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, default=None)
    model_group.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--scaling-preset", action="store_true")
    parser.add_argument("--tasks", type=str, default="all",
                        choices=["all", "single", "chain", "adversarial"])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/sprint1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-epochs", type=int, default=50)

    args = parser.parse_args()

    if args.scaling_preset:
        args.model_list = SCALING_MODELS
    elif args.models:
        args.model_list = args.models
    elif args.model:
        args.model_list = [args.model]
    else:
        args.model_list = ["Qwen/Qwen3-8B"]

    return args


def model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def run_sprint1_for_model(model_id: str, sprint0_dir: Path, output_dir: Path, args):
    """Sprint 1 complet pour un modèle : train + benchmark A/B."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from baselines.react_agent import ReActAgent, HiddenStateLogger
    from baselines.tools import TOOL_MAP, get_tool_names
    from baselines.tasks import ALL_TASKS, SINGLE_TASKS, CHAIN_TASKS, ADVERSARIAL_TASKS
    from baselines.failure_analysis import analyze_traces, print_report

    short = model_short_name(model_id)
    sprint0_model_dir = sprint0_dir / short
    model_output_dir = output_dir / short
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'╔' + '═' * 68 + '╗'}")
    logger.info(f"║ SPRINT 1 : {model_id}")
    logger.info(f"{'╚' + '═' * 68 + '╝'}")

    # Vérifier que Sprint 0 existe
    if not sprint0_model_dir.exists():
        logger.error(f"Sprint 0 results not found: {sprint0_model_dir}")
        return None

    # ── Phase 1 : Charger Â detector ──
    logger.info("═══ Phase 1 : Â Detector ═══")
    a_detector = AHatDetector.from_sprint0(sprint0_model_dir)

    # ── Phase 2 : Entraîner l'embedding contrastif ──
    logger.info("═══ Phase 2 : Contrastive Tool Embedding ═══")

    # Charger les traces Sprint 0
    with open(sprint0_model_dir / "traces.json") as f:
        sprint0_traces = json.load(f)

    # Détecter hidden_dim depuis les traces
    hidden_dim = None
    for trace in sprint0_traces:
        for step in trace.get("steps", []):
            hs_path = step.get("hidden_state_path")
            if hs_path and Path(hs_path).exists():
                h = np.load(hs_path)
                hidden_dim = h.shape[0]
                break
        if hidden_dim:
            break

    if hidden_dim is None:
        logger.error("No hidden states found in Sprint 0 traces")
        return None

    tool_embedder = ContrastiveToolEmbedding(hidden_dim=hidden_dim)
    embed_results = tool_embedder.train_from_traces(
        traces=sprint0_traces,
        hidden_states_dir=sprint0_model_dir / "hidden_states",
        tool_names=get_tool_names(),
        epochs=args.contrastive_epochs,
        device=args.device,
    )
    logger.info(f"Contrastive embedding: {embed_results}")

    if "error" not in embed_results:
        tool_embedder.save(model_output_dir / "tool_embedding")

    # ── Phase 3 : Charger le modèle et benchmarker ──
    logger.info("═══ Phase 3 : Benchmark A/B ═══")
    logger.info(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else 32

    # Sélectionner les tâches
    if args.tasks == "single":
        tasks = SINGLE_TASKS
    elif args.tasks == "chain":
        tasks = CHAIN_TASKS
    elif args.tasks == "adversarial":
        tasks = ADVERSARIAL_TASKS
    else:
        tasks = ALL_TASKS

    # ── Condition A : Baseline textuel (rappel Sprint 0) ──
    logger.info(f"\n─── Condition A : Baseline textuel ({len(tasks)} tâches) ───")
    hs_logger_a = HiddenStateLogger(model, num_layers, model_output_dir / "hs_baseline")
    agent_a = ReActAgent(
        model=model, tokenizer=tokenizer, tools=TOOL_MAP,
        hs_logger=hs_logger_a, max_steps=args.max_steps,
        temperature=args.temperature, device=args.device,
    )

    traces_a = []
    for i, task in enumerate(tasks):
        logger.info(f"  A [{i+1}/{len(tasks)}] {task.id}...")
        trace = agent_a.run(task.id, task.prompt, task.category, task.expected_tools)
        traces_a.append(trace)
        status = "✓" if trace.final_answer else "✗"
        logger.info(f"    {status} steps={len(trace.steps)} tools={trace.tools_used}")
    hs_logger_a.cleanup()

    # ── Condition B : Â gating + routing contrastif ──
    logger.info(f"\n─── Condition B : Â + routing contrastif ({len(tasks)} tâches) ───")
    hs_logger_b = HiddenStateLogger(model, num_layers, model_output_dir / "hs_sprint1")
    logits_router = LogitsToolRouter(tokenizer, get_tool_names())

    agent_b = Sprint1Agent(
        model=model, tokenizer=tokenizer, tools=TOOL_MAP,
        a_detector=a_detector, tool_router=tool_embedder,
        logits_router=logits_router, hs_logger=hs_logger_b,
        max_steps=args.max_steps, temperature=args.temperature,
        device=args.device, routing_mode="contrastive", a_hat_enabled=True,
    )

    traces_b = []
    for i, task in enumerate(tasks):
        logger.info(f"  B [{i+1}/{len(tasks)}] {task.id}...")
        trace = agent_b.run(task.id, task.prompt, task.category, task.expected_tools)
        traces_b.append(trace)
        status = "✓" if trace.final_answer else "✗"
        logger.info(f"    {status} steps={len(trace.steps)} tools={trace.tools_used}")
    hs_logger_b.cleanup()

    # ── Analyse comparative ──
    logger.info(f"\n{'═' * 70}")
    logger.info("RÉSULTATS A/B")
    logger.info("═" * 70)

    analysis_a = analyze_traces(traces_a, tasks)
    analysis_b = analyze_traces(traces_b, tasks)

    print(f"\n{'=' * 70}")
    print(f"CONDITION A (baseline textuel) :")
    print(f"  Success rate : {analysis_a['success_rate']:.1%}")
    print(f"  Dominant failure : {analysis_a['dominant_failure']}")
    print(f"  Mean steps : {analysis_a['step_stats']['mean_steps']:.1f}")

    print(f"\nCONDITION B (Â gating + routing contrastif) :")
    print(f"  Success rate : {analysis_b['success_rate']:.1%}")
    print(f"  Dominant failure : {analysis_b['dominant_failure']}")
    print(f"  Mean steps : {analysis_b['step_stats']['mean_steps']:.1f}")

    delta = analysis_b['success_rate'] - analysis_a['success_rate']
    print(f"\nΔ SUCCESS RATE : {delta:+.1%}")

    if delta > 0.05:
        print(f"  ✓ GAIN SIGNIFICATIF — le routing géométrique améliore l'agent")
    elif delta > 0.02:
        print(f"  ~ Gain marginal — le routing aide mais faiblement")
    elif delta > -0.02:
        print(f"  = Pas de différence significative")
    else:
        print(f"  ✗ RÉGRESSION — le routing géométrique nuit")

    # No-tool comparison
    notool_a = analysis_a['failure_modes'].get('no_tool', {}).get('rate', 0)
    notool_b = analysis_b['failure_modes'].get('no_tool', {}).get('rate', 0)
    if isinstance(notool_a, str): notool_a = 0
    if isinstance(notool_b, str): notool_b = 0
    print(f"\nNo-tool rate : A={notool_a:.1%} → B={notool_b:.1%} (Δ={notool_b-notool_a:+.1%})")
    print("=" * 70)

    # Sauvegarder
    results = {
        "model_id": model_id,
        "condition_a": {"success_rate": analysis_a["success_rate"],
                       "dominant_failure": analysis_a["dominant_failure"]},
        "condition_b": {"success_rate": analysis_b["success_rate"],
                       "dominant_failure": analysis_b["dominant_failure"]},
        "delta_success": float(delta),
        "delta_no_tool": float(notool_b - notool_a),
        "contrastive_embedding": embed_results,
    }
    with open(model_output_dir / "sprint1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Sauvegarder les traces
    with open(model_output_dir / "traces_baseline.json", "w") as f:
        json.dump([t.to_dict() for t in traces_a], f, indent=2)
    with open(model_output_dir / "traces_sprint1.json", "w") as f:
        json.dump([t.to_dict() for t in traces_b], f, indent=2)

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()
    sprint0_dir = Path(args.sprint0_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SPRINT 1 — Â DETECTOR + ROUTING GÉOMÉTRIQUE")
    logger.info("=" * 70)
    logger.info(f"Sprint 0 dir : {sprint0_dir}")
    logger.info(f"Models : {args.model_list}")
    logger.info(f"Output : {output_dir}")

    all_results = {}

    for model_id in args.model_list:
        results = run_sprint1_for_model(model_id, sprint0_dir, output_dir, args)
        if results:
            all_results[model_id] = results

    # Comparaison cross-modèle
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("SCALING ANALYSIS — SPRINT 1")
        print("=" * 70)
        print(f"{'Model':<25s} {'A (base)':>10s} {'B (geo)':>10s} {'Δ':>8s} {'Δ no_tool':>10s}")
        print("-" * 65)
        for model_id, r in sorted(all_results.items(),
                                    key=lambda x: x[1].get("delta_success", 0)):
            a = r["condition_a"]["success_rate"]
            b = r["condition_b"]["success_rate"]
            d = r["delta_success"]
            dn = r["delta_no_tool"]
            print(f"{model_short_name(model_id):<25s} {a:>9.1%} {b:>9.1%} {d:>+7.1%} {dn:>+9.1%}")
        print("=" * 70)

    with open(output_dir / "sprint1_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nSprint 1 complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
