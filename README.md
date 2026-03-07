# JEPA-Agent — Sprint 0

Agent IA fondé sur Joint Embedding Predictive Architecture.
Sprint 0 : baseline + diagnostic + validation Â + **analyse de scaling géométrique**.

## Quick Start (RunPod A100 80GB)

```bash
# Setup
pip install -r requirements.txt
huggingface-cli login  # Token HF pour accès modèles gated

# ─── Single model (comme avant) ───
python run_sprint0.py --model Qwen/Qwen3-8B

# ─── Multi-model scaling analysis (recommandé) ───

# Preset : 1.5B + 3B + 7B (même famille, isole l'effet taille)
python run_sprint0.py --scaling-preset

# Custom : choisir les modèles
python run_sprint0.py --models Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B

# Maximum delta : ajouter le 0.5B
python run_sprint0.py --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B

# ─── Options ───
# Test rapide (single tasks seulement)
python run_sprint0.py --scaling-preset --tasks single

# Avec R̂ pré-calculés pour comparaison cos(Â, R̂)
python run_sprint0.py --scaling-preset --r-hat-dir r_hat/

# Sans extraction Â (0A + 0B seulement)
python run_sprint0.py --scaling-preset --skip-0c
```

## VRAM par modèle (bf16)

| Modèle | VRAM | Note |
|--------|------|------|
| Qwen3-0.6B | ~1 GB | Plancher, intéressant pour le delta max |
| Qwen3-1.7B | ~3 GB | Bon compromis signal/taille |
| Qwen3-4B | ~6 GB | Point intermédiaire |
| Qwen3-8B | ~14 GB | Reference principale |
| Qwen3-14B | ~28 GB | Optionnel, si budget le permet |

Les modèles sont chargés et déchargés séquentiellement — un seul en VRAM à la fois.

## Outputs

```
results/sprint0/
├── config.json                          # Configuration du run
├── scaling_analysis.json                # Analyse comparative (Sprint 0D)
├── qwen2.5-1.5b-instruct/              # Par modèle
│   ├── traces.json                      # Traces complètes
│   ├── failure_analysis.json            # Diagnostic d'échecs (0B)
│   ├── a_hat_traces.json                # Résultats Â (0C)
│   ├── a_hat_extracted.npy              # Direction Â extraite
│   ├── summary.json                     # Résumé du modèle
│   └── hidden_states/                   # .npy par step
├── qwen2.5-3b-instruct/
│   └── ...
└── qwen3-8b/
    └── ...
```

## Structure

```
jepa-agent/
├── baselines/
│   ├── tools.py              # 25 tools, 5 catégories
│   ├── executors.py          # Exécution réelle (15 real + 5 API + 5 mock)
│   ├── tasks.py              # 60 tâches (40 single, 12 chain, 8 adversarial)
│   ├── react_agent.py        # Agent ReAct + hidden state logger
│   └── failure_analysis.py   # Taxonomie d'échecs Sprint 0B
├── geometry/                  # Sprint 0C + Sprint 1 (à venir)
├── run_sprint0.py            # Point d'entrée (multi-model)
└── requirements.txt
```
