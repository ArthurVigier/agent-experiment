# RunPod Setup — Sprint 0 JEPA-Agent

Guide concret, pas à pas. Du compte RunPod au premier résultat.

---

## 1. Choix du hardware

### Quel GPU ?

La décision dépend de ce que tu veux tester :

| Scénario | GPU | VRAM | Coût/h | Durée Sprint 0 | Budget total |
|---|---|---|---|---|---|
| **Single model (7B)** | 1× A100 80GB SXM | 80 GB | ~$1.64 | 2-3h | **$3-5** |
| **Scaling preset (1.5B+3B+7B)** | 1× A100 80GB SXM | 80 GB | ~$1.64 | 4-6h | **$7-10** |
| **Full scaling (0.5B+1.5B+3B+7B)** | 1× A100 80GB SXM | 80 GB | ~$1.64 | 5-8h | **$8-13** |
| **Test rapide** | 1× RTX 4090 | 24 GB | ~$0.44 | 1-2h (single only) | **$1-2** |
| **Budget minimal** | 1× L40 / A40 | 48 GB | ~$0.80 | 3-5h | **$3-4** |

**Recommandation : A100 80GB SXM, On-Demand.** C'est le sweet spot. 80GB de VRAM signifie que tu charges le 7B en bf16 (15GB), il reste 65GB pour les hidden states en mémoire et les éventuels peaks. Et c'est assez pour charger des modèles jusqu'à 32B si tu veux pousser le scaling plus tard.

**Pourquoi On-Demand et pas Spot ?** Sprint 0 dure 2-6h selon le nombre de modèles. Un preempt Spot interrompt le run, tu perds les traces en cours, et tu dois tout relancer. L'économie (~30%) ne vaut pas le risque sur un run de cette durée.

### VRAM par modèle (bf16)

| Modèle | VRAM bf16 | VRAM GPTQ-4bit | Note |
|---|---|---|---|
| Qwen3-0.6B | ~1.2 GB | ~0.5 GB | Plancher, trivial |
| Qwen3-1.7B | ~3.5 GB | ~1.5 GB | Léger |
| Qwen3-4B | ~7 GB | ~3 GB | Confortable |
| Qwen3-8B | ~15 GB | ~5 GB | Reference |
| Qwen3-14B | ~29 GB | ~10 GB | Optionnel, gros |
| Qwen3-32B | ~65 GB | ~20 GB | Tight sur A100 80GB |

Les modèles sont chargés et déchargés **séquentiellement** — un seul en VRAM à la fois. Le 0.5B + 1.5B + 3B + 7B = **pas 26GB en parallèle**, mais 15GB max à tout moment (le 7B est le plus gros).

### Configuration complète du pod

| Paramètre | Valeur | Pourquoi |
|---|---|---|
| **GPU** | 1× NVIDIA A100 80GB SXM | Voir tableau ci-dessus |
| **GPU Count** | 1 | Suffisant pour Sprint 0 |
| **Template** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | PyTorch + CUDA pré-installé |
| **Container Disk** | 50 GB | Espace pour pip, cache transformers |
| **Volume Disk** | 100 GB | Modèles HF (~15GB/modèle) + hidden states (~5-15GB) + marge |
| **Volume Mount** | `/workspace` | **PERSISTANT** entre stop/restart |
| **vCPU** | 8+ | Preprocessing, pas critique |
| **RAM** | 62 GB (ou max dispo) | Hidden states cumulés tiennent en RAM |
| **Type** | **On-Demand** | Pas Spot — preempt = perte du run |

**Le volume à 100GB** : c'est plus que les 50GB du guide précédent. En mode scaling avec 4 modèles, les hidden states + modèles cachés HF prennent ~40-60GB. 100GB donne de la marge.

---

## 2. Création du Pod — pas à pas

### 2a. Pré-requis

1. Compte RunPod : https://www.runpod.io/
2. Crédit ajouté ($25 minimum recommandé pour Sprint 0 + marge)
3. **Clé SSH** : Settings → SSH Keys → ajouter ta clé publique (`cat ~/.ssh/id_rsa.pub`)

### 2b. Déployer

1. Dashboard → **Pods** → **+ Deploy**
2. Sélectionner **GPU Pod**
3. **Filtrer** : cocher "80 GB" VRAM, trier par prix
4. Sélectionner **A100 80GB SXM** (le moins cher disponible)
5. **Customize Deployment** :
   - Template : chercher `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   - Container Disk : 50 GB
   - Volume Disk : 100 GB
   - Volume Mount Path : `/workspace`
6. **Deploy On-Demand**
7. Attendre ~1-3 min que le pod soit "Running"

### 2c. Se connecter

**SSH (recommandé)** :
```bash
# La commande est affichée dans le dashboard du pod
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
```

**Web Terminal** : Dashboard → cliquer sur le pod → "Connect" → "Start Web Terminal"

**Jupyter** : Le template PyTorch expose aussi un Jupyter Lab sur le port 8888. Utile pour inspecter les résultats interactivement.

---

## 3. Setup de l'environnement

Tout se passe dans `/workspace` (volume persistant). Ne PAS travailler dans `/root` (container disk, non persistant).

```bash
# ═══ Se placer dans le volume persistant ═══
cd /workspace

# ═══ Uploader le projet ═══

# Option A : scp depuis ta machine locale
# (sur ta machine, PAS sur le pod)
# scp -P <port> jepa-agent-sprint0.tar.gz root@<pod-ip>:/workspace/

# Option B : wget si hébergé quelque part
# wget <url>/jepa-agent-sprint0.tar.gz

# Extraire
tar xzf jepa-agent-sprint0.tar.gz
cd jepa-agent

# ═══ Installer les dépendances ═══
pip install -r requirements.txt

# Vérifier l'installation
pip list | grep -E "torch|transformers|accelerate|duckduckgo|sklearn"
# Attendu :
# torch                    2.x.x
# transformers             4.4x.x
# accelerate               0.x.x
# duckduckgo-search        6.x.x
# scikit-learn             1.x.x
```

### HuggingFace login

Les modèles Qwen3 sont sous Apache 2.0, **pas** gated — pas besoin de login pour eux. Mais si tu veux tester des modèles gated (Llama 3, Mistral), il faut un token :

```bash
pip install huggingface_hub
huggingface-cli login
# → Coller ton token depuis https://huggingface.co/settings/tokens
# → Accepter les conditions d'utilisation du modèle sur la page HF
```

### Vérifier CUDA et GPU

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_mem / 1e9:.1f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
"
# Attendu :
# PyTorch: 2.4.x
# CUDA available: True
# GPU: NVIDIA A100-SXM4-80GB
# VRAM: 80.0 GB
# Compute capability: 8.0
```

### Préparer le sandbox de fichiers

Les tools fichier (read_file, csv_analyze, etc.) cherchent dans `/tmp/jepa_sandbox/`. Il faut y créer des fichiers réalistes pour que les tâches fonctionnent :

```bash
mkdir -p /tmp/jepa_sandbox/data

# CSV avec des vraies données
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

# Fichier texte
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

# Script Python
cat > /tmp/jepa_sandbox/test_script.py << 'EOF'
def main():
    print('hello from test file')

if __name__ == '__main__':
    main()
EOF
```

### Vérification complète

```bash
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

# Test un executor réel
result = EXECUTOR_MAP['calculator']('2**32')
print(f'Calculator test: 2^32 = {result}')
result = EXECUTOR_MAP['read_file']('results.csv')
print(f'Read file test: {result[:80]}...')
print()
print('✓ READY TO GO')
"
```

Si tu vois `✓ READY TO GO`, tout est prêt.

---

## 4. Lancer Sprint 0

### Étape 1 : Test rapide (5-10 min)

**Toujours faire un test rapide d'abord.** Ça valide que le modèle se charge, que l'agent parse les tool calls, et que les executors tournent.

```bash
python run_sprint0.py \
    --model Qwen/Qwen3-8B \
    --tasks single \
    --max-steps 5 \
    --output-dir results/sprint0_test \
    --skip-0c
```

**Ce qui doit se passer :**
1. Le modèle se charge (~1 min, première fois ~5 min pour le téléchargement)
2. Sprint 0-PRE tourne en < 30 secondes → affiche la silhouette des embeddings
3. Sprint 0A lance 40 tâches single → chaque tâche affiche ✓ ou ✗ avec le temps
4. Sprint 0B affiche le rapport d'échecs

**Signaux d'alerte :**
- OOM au chargement → voir Troubleshooting
- 0 tool calls sur les 10 premières tâches → le modèle ne parse pas le format ReAct → voir Troubleshooting
- Toutes les tasks < 2 secondes → le modèle génère des réponses vides

### Étape 2 : Run complet single model (2-3h)

```bash
python run_sprint0.py \
    --model Qwen/Qwen3-8B \
    --output-dir results/sprint0
```

### Étape 3 : Run complet multi-modèle scaling (4-6h)

```bash
# Preset : 1.5B + 3B + 7B (recommandé)
python run_sprint0.py \
    --scaling-preset \
    --output-dir results/sprint0

# OU avec le 0.5B pour max delta (5-8h)
python run_sprint0.py \
    --models \
        Qwen/Qwen3-0.6B \
        Qwen/Qwen3-1.7B \
        Qwen/Qwen3-4B \
        Qwen/Qwen3-8B \
    --output-dir results/sprint0
```

### Étape 4 : Avec R̂ pré-calculé (optionnel)

Si tu as le R̂ de ton paper register-geometry :

```bash
# Uploader les R̂ depuis ta machine
mkdir -p /workspace/jepa-agent/r_hat
# scp -P <port> r_hat_*.npy root@<pod-ip>:/workspace/jepa-agent/r_hat/
# Nommer les fichiers : qwen2.5-7b-instruct.npy, qwen2.5-3b-instruct.npy, etc.

python run_sprint0.py \
    --scaling-preset \
    --r-hat-dir r_hat/ \
    --output-dir results/sprint0
```

### Lancer en arrière-plan (recommandé pour les runs longs)

```bash
# Avec nohup — le run continue même si tu te déconnectes
nohup python run_sprint0.py \
    --scaling-preset \
    --output-dir results/sprint0 \
    > sprint0_stdout.log 2>&1 &

# Récupérer le PID
echo $!
# → 12345

# Vérifier que ça tourne
ps aux | grep run_sprint0
```

### Monitoring pendant l'exécution

```bash
# Terminal 1 : VRAM en temps réel
watch -n 5 nvidia-smi

# Terminal 2 : Logs
tail -f sprint0.log

# Terminal 3 : Progress
watch -n 30 'ls -la results/sprint0/*/traces.json 2>/dev/null'
```

---

## 5. Interpréter les résultats rapides

### Sprint 0-PRE (immédiat, pendant le chargement)

```
SPRINT 0-PRE : STRUCTURE FONCTIONNELLE DANS LES EMBEDDINGS
═══════════════════════════════════════════════════════════
  Silhouette moyenne : 0.XXX
```

- **> 0.30** : excellent. L'espace fonctionnel existe dans les poids. Le routing géométrique a une base solide.
- **0.10 - 0.30** : partiel. Il y a du signal mais c'est bruité. À utiliser comme initialisation.
- **< 0.10** : pas de structure dans les embeddings. Normal pour certains modèles — ça ne veut pas dire que les hidden states n'ont pas de structure.

### Sprint 0B (après 0A, le diagnostic clé)

```
Mode d'échec dominant : wrong_tool
```

C'est ce qui détermine Sprint 1. Regarde le pourcentage :
- `wrong_tool > 30%` → Sprint 1A (routing géométrique)
- `loop/no_tool > 30%` → Sprint 1B (Â detector)
- `premature_stop > 30%` → Sprint 1C (trajectory planner)

### Sprint 0C (si pas --skip-0c)

```
AUC(R̂ → tool call) = 0.XXX
```

- **> 0.75** : R̂ prédit les tool calls. Â detector initialisable avec R̂.
- **0.60 - 0.75** : signal partiel. Probe supervisé nécessaire.
- **< 0.60** : cascade de fallback lancée automatiquement.

### Sprint 0D (multi-modèle seulement)

Le tableau comparatif et le verdict scaling. C'est potentiellement le résultat le plus important de tout le Sprint 0.

---

## 6. Récupérer les résultats

### Structure des outputs

```
results/sprint0/
├── config.json                          # Config du run
├── scaling_analysis.json                # Sprint 0D (si multi-modèle)
│
├── qwen2.5-0.5b-instruct/              # Par modèle
│   ├── embedding_structure.json         # Sprint 0-PRE ← PETIT, IMPORTANT
│   ├── tool_embeddings.npy              # Embeddings pour réutilisation
│   ├── tool_similarity_matrix.npy       # Matrice de similarité
│   ├── traces.json                      # Sprint 0A ← MOYEN, CRITIQUE
│   ├── failure_analysis.json            # Sprint 0B ← PETIT, CRITIQUE
│   ├── a_hat_traces.json                # Sprint 0C ← PETIT, IMPORTANT
│   ├── a_hat_extracted.npy              # Direction Â
│   ├── summary.json                     # Résumé
│   └── hidden_states/                   # ← GROS (2-10GB par modèle)
│       ├── s01_step000.npy
│       └── ...
│
├── qwen2.5-1.5b-instruct/
│   └── ...
├── qwen2.5-3b-instruct/
│   └── ...
└── qwen2.5-7b-instruct/
    └── ...
```

### Télécharger (par ordre de priorité)

```bash
# 1. Les JSON d'abord (quelques KB, les résultats essentiels)
cd /workspace/jepa-agent
for dir in results/sprint0/*/; do
    echo "=== $dir ==="
    ls -lh "$dir"/*.json 2>/dev/null
done

# Compresser les JSON seulement
find results/sprint0 -name "*.json" | tar czf /workspace/sprint0_json.tar.gz -T -

# 2. Les .npy (embeddings + Â extrait, quelques MB)
find results/sprint0 -name "*.npy" ! -path "*/hidden_states/*" | tar czf /workspace/sprint0_npy.tar.gz -T -

# 3. Les hidden states (GROS, 2-10GB par modèle)
# Seulement si tu veux faire Sprint 0C-bis ou Sprint 1 localement
# Sinon, les garder sur le volume RunPod
tar czf /workspace/sprint0_hidden_states.tar.gz results/sprint0/*/hidden_states/
```

```bash
# Sur ta machine locale :
scp -P <port> root@<pod-ip>:/workspace/sprint0_json.tar.gz .
scp -P <port> root@<pod-ip>:/workspace/sprint0_npy.tar.gz .
# Les hidden states seulement si nécessaire (GROS transfert)
```

---

## 7. Après Sprint 0

### STOP le pod, ne le TERMINATE pas

- **Stop** : GPU libéré, plus de coût GPU. Volume persistant (~$0.10/GB/mois ≈ $10/mois pour 100GB)
- **Terminate** : TOUT est détruit. Les modèles téléchargés, les hidden states, tout.

Le volume contient :
- Cache HuggingFace (`~/.cache/huggingface/`) : ~15-60GB selon les modèles téléchargés
- Hidden states : ~5-30GB selon le nombre de modèles
- Résultats : quelques MB

**Tu relanceras le même pod pour Sprint 1.** Tout sera là.

### Nettoyage optionnel (réduire le coût du volume)

```bash
# Supprimer les hidden states des modèles qui ne t'intéressent pas
rm -rf results/sprint0/qwen2.5-0.5b-instruct/hidden_states/

# Supprimer le cache des modèles déchargés (re-téléchargement nécessaire)
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/

# Vérifier l'espace
du -sh /workspace/* | sort -h
```

---

## 8. Troubleshooting

### OOM (Out of Memory)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Causes et solutions :**

1. **Un autre modèle est encore en VRAM** — le garbage collector n'a pas libéré
   ```bash
   # Vérifier
   nvidia-smi
   # Si VRAM > 0 sans process : restart le kernel Python
   ```

2. **Le modèle est trop gros** — 7B en bf16 = 15GB, ça passe sur 24GB+ facilement
   ```bash
   # Utiliser la version quantisée
   pip install auto-gptq optimum
   python run_sprint0.py --model Qwen/Qwen3-8B-GPTQ-Int4
   ```

3. **Accumulation de hidden states en VRAM** — les tenseurs detach() devraient être en CPU
   ```python
   # Vérifier dans un shell Python
   import torch
   print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   print(f"VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
   ```

### Le modèle ne fait aucun tool call

**Symptôme** : toutes les tâches finissent en `max_steps_reached`, 0 tool calls dans les traces.

**Diagnostic rapide** :
```python
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
```

**Si l'output ne contient pas `Action: web_search(...)` :**
- Le modèle ne suit pas le format ReAct → essayer un modèle plus gros (le 1.5B est souvent trop petit pour suivre des instructions complexes)
- Réduire le system prompt (le prompt actuel avec la description de 25 tools est long)
- Augmenter la température à 0.3

### DuckDuckGo rate limité

```
duckduckgo_search.exceptions.RatelimitException
```

**Solution** : ajouter un délai entre les requêtes. Dans `baselines/executors.py`, ajouter `time.sleep(2)` au début de `api_web_search`. Ou ignorer (le fallback retourne un message d'erreur, l'agent continue).

### Les fichiers sandbox ne sont pas trouvés

```
ERROR: File not found: 'results.csv'
```

Le sandbox est dans `/tmp/jepa_sandbox/`. Vérifier que le step 3 (préparation du sandbox) a bien été exécuté. Le `/tmp` est vidé au restart du pod — **relancer le setup du sandbox à chaque restart**.

### Le run a planté au milieu

Les traces sont sauvegardées au fur et à mesure dans `traces.json`. Si le run plante au step 45/60, les 44 premières traces sont sauvegardées. Tu peux :

```bash
# Voir combien de traces ont été sauvegardées
python -c "import json; d=json.load(open('results/sprint0/qwen2.5-7b-instruct/traces.json')); print(f'{len(d)} traces')"

# Relancer seulement les tâches manquantes (pas encore implémenté, mais les traces existantes sont exploitables)
# Sprint 0B et 0C fonctionnent sur des traces partielles
```

---

## 9. Checklist avant le lancement

```
PRÉ-REQUIS
  [ ] Compte RunPod avec $25+ de crédit
  [ ] Clé SSH ajoutée dans RunPod Settings

DÉPLOIEMENT
  [ ] Pod A100 80GB SXM déployé (On-Demand)
  [ ] Status : Running
  [ ] SSH connecté

SETUP
  [ ] cd /workspace
  [ ] Projet extrait dans /workspace/jepa-agent
  [ ] pip install -r requirements.txt
  [ ] nvidia-smi → GPU visible
  [ ] torch.cuda.is_available() → True
  [ ] Fichiers sandbox créés (/tmp/jepa_sandbox/)
  [ ] Vérification complète : "✓ READY TO GO"

TEST RAPIDE
  [ ] python run_sprint0.py --model Qwen/Qwen3-8B --tasks single --max-steps 5 --skip-0c
  [ ] Sprint 0-PRE affiche la silhouette
  [ ] L'agent fait des tool calls (pas 0 tool calls partout)
  [ ] Pas d'OOM

LANCEMENT
  [ ] nohup python run_sprint0.py --scaling-preset --output-dir results/sprint0 > stdout.log 2>&1 &
  [ ] watch nvidia-smi → le GPU travaille
  [ ] tail -f sprint0.log → les tâches défilent

APRÈS
  [ ] Résultats JSON téléchargés
  [ ] Pod STOPPÉ (pas terminé)
  [ ] Résultats inspectés localement
```

---

## 10. Estimation du temps détaillée

### Single model (Qwen3-8B)

| Étape | Durée | Détail |
|---|---|---|
| Chargement modèle | 1-5 min | 1 min si caché, 5 min premier téléchargement |
| **0-PRE** | < 30 sec | Lookup embeddings, pas de forward pass |
| **0A** (60 tâches) | 1.5-3h | ~1-3 min/tâche (dépend du nombre de steps) |
| **0B** | < 1 min | Analyse des traces, pas de GPU |
| **0C** | 2-5 min | Projections sur les hidden states déjà sauvegardés |
| **Total** | **~2-3h** | |

### Scaling preset (1.5B + 3B + 7B)

| Étape | Durée | Détail |
|---|---|---|
| Qwen3-1.7B | 30-60 min | Petit modèle, génération rapide |
| Déchargement + chargement 3B | 2-3 min | |
| Qwen3-4B | 45-90 min | |
| Déchargement + chargement 7B | 2-5 min | |
| Qwen3-8B | 1.5-3h | Le plus lent |
| **0D** (analyse comparative) | < 1 min | |
| **Total** | **~3.5-6h** | |

### Full scaling (0.5B + 1.5B + 3B + 7B)

Ajouter ~20-40 min pour le 0.5B. **Total : ~4-7h.**
