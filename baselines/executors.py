"""
executors.py
Exécuteurs réels pour les tools Sprint 0.

Trois catégories :
  - REAL : exécution réelle (Python, filesystem, calcul)
  - API  : appels à des APIs gratuites (search, arxiv, weather, fetch)
  - MOCK : simulation réaliste pour les tools à effets de bord (email, slack, calendar)

Les mocks retournent des formats identiques aux vraies APIs
(Slack JSON, email headers, etc.) pour que les hidden states
de l'agent soient dans un état sémantique réaliste.

Convention : docstrings français, code anglais.
"""

import ast
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
import logging
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Config ─────────────────────────────────────────────────────────────────

# Répertoire sandbox pour les opérations fichier
SANDBOX_DIR = Path(os.environ.get("JEPA_SANDBOX_DIR", "/tmp/jepa_sandbox"))
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

# Timeout pour les exécutions de code (secondes)
CODE_TIMEOUT = 10

# APIs (clés optionnelles, les tools fonctionnent sans mais avec moins de qualité)
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════════════
# REAL EXECUTORS — exécution authentique
# ═══════════════════════════════════════════════════════════════════════════

def real_python_execute(code: str) -> str:
    """
    Exécute du code Python dans un subprocess sandboxé.
    Capture stdout, stderr, et retourne le résultat.
    """
    # Écrire le code dans un fichier temporaire
    code_file = SANDBOX_DIR / f"_exec_{uuid.uuid4().hex[:8]}.py"
    code_file.write_text(code)

    try:
        result = subprocess.run(
            [sys.executable, str(code_file)],
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
            cwd=str(SANDBOX_DIR),
            env={**os.environ, "PYTHONPATH": str(SANDBOX_DIR)},
        )
        output = ""
        if result.stdout.strip():
            output += result.stdout.strip()
        if result.stderr.strip():
            if output:
                output += "\n"
            output += f"[stderr]: {result.stderr.strip()}"
        if not output:
            output = "Code executed successfully. No output."
        return output
    except subprocess.TimeoutExpired:
        return f"ERROR: Execution timed out after {CODE_TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    finally:
        code_file.unlink(missing_ok=True)


def real_python_eval(expression: str) -> str:
    """
    Évalue une expression Python de manière sécurisée.
    Autorise les fonctions mathématiques et les builtins safe.
    """
    allowed_builtins = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "len": len, "int": int, "float": float,
        "str": str, "bool": bool, "list": list, "tuple": tuple,
        "range": range, "enumerate": enumerate, "zip": zip,
        "sorted": sorted, "reversed": reversed, "map": map, "filter": filter,
        "pow": pow, "divmod": divmod, "hex": hex, "bin": bin, "oct": oct,
        "True": True, "False": False, "None": None,
    }
    math_funcs = {
        name: getattr(math, name)
        for name in dir(math)
        if not name.startswith("_")
    }
    namespace = {**allowed_builtins, **math_funcs}

    try:
        # Vérifier que c'est une expression (pas un statement)
        ast.parse(expression, mode="eval")
        result = eval(expression, {"__builtins__": {}}, namespace)
        return str(result)
    except SyntaxError:
        # Si c'est un statement, l'exécuter
        try:
            stdout_capture = io.StringIO()
            exec_globals = {"__builtins__": allowed_builtins, **math_funcs}
            with redirect_stdout(stdout_capture):
                exec(expression, exec_globals)
            output = stdout_capture.getvalue().strip()
            return output if output else "Executed successfully. No output."
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def real_calculator(expression: str) -> str:
    """Évalue une expression mathématique."""
    return real_python_eval(expression)


def real_shell_command(command: str) -> str:
    """
    Exécute une commande shell sandboxée.
    Bloque les commandes dangereuses.
    """
    # Blocklist de commandes dangereuses
    dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/", "chmod -R 777 /",
                  "curl | sh", "wget | sh", "sudo", "shutdown", "reboot"]
    cmd_lower = command.lower()
    for d in dangerous:
        if d in cmd_lower:
            return f"ERROR: Command blocked for safety: '{command}'"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
            cwd=str(SANDBOX_DIR),
        )
        output = ""
        if result.stdout.strip():
            output += result.stdout.strip()
        if result.stderr.strip():
            if output:
                output += "\n"
            output += f"[stderr]: {result.stderr.strip()}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output if output else "Command executed. No output."
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {CODE_TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def real_install_package(package_name: str) -> str:
    """Installe un package Python via pip (dans le sandbox)."""
    # Sécurité minimale
    if not re.match(r'^[a-zA-Z0-9_\-\[\]>=<.,]+$', package_name):
        return f"ERROR: Invalid package name: '{package_name}'"

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package_name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            # Parser la version installée
            output = result.stdout.strip()
            if not output:
                output = f"Successfully installed {package_name}"
            return output
        else:
            return f"ERROR: pip install failed:\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return f"ERROR: Installation timed out"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def real_read_file(filepath: str) -> str:
    """Lit un fichier depuis le sandbox."""
    # Résoudre le chemin relatif au sandbox
    path = _resolve_sandbox_path(filepath)

    if not path.exists():
        return f"ERROR: File not found: '{filepath}'"
    if not path.is_file():
        return f"ERROR: '{filepath}' is not a file"
    if path.stat().st_size > 1_000_000:  # 1MB max
        return f"ERROR: File too large ({path.stat().st_size} bytes). Max: 1MB."

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        # Tronquer si très long
        if len(content) > 10_000:
            return content[:10_000] + f"\n\n[... truncated, {len(content)} chars total]"
        return content
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def real_write_file(filepath: str, content: str) -> str:
    """Écrit du contenu dans un fichier dans le sandbox."""
    path = _resolve_sandbox_path(filepath)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {filepath}."
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def real_list_directory(path: str = ".") -> str:
    """Liste le contenu d'un répertoire dans le sandbox."""
    dir_path = _resolve_sandbox_path(path)

    if not dir_path.exists():
        return f"ERROR: Directory not found: '{path}'"
    if not dir_path.is_dir():
        return f"ERROR: '{path}' is not a directory"

    try:
        entries = sorted(dir_path.iterdir())
        lines = []
        for entry in entries[:50]:  # Max 50 entries
            if entry.is_dir():
                lines.append(f"  {entry.name}/ (directory)")
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/1024/1024:.1f}MB"
                lines.append(f"  {entry.name} ({size_str})")
        if len(entries) > 50:
            lines.append(f"  ... and {len(entries) - 50} more entries")
        header = f"Contents of {path}/:" if lines else f"{path}/ is empty"
        return header + "\n" + "\n".join(lines)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def real_csv_analyze(filepath: str, operation: str = "describe") -> str:
    """Analyse un CSV avec pandas (installé à la demande)."""
    path = _resolve_sandbox_path(filepath)
    if not path.exists():
        return f"ERROR: File not found: '{filepath}'"

    try:
        import pandas as pd
    except ImportError:
        return "ERROR: pandas not installed. Call install_package('pandas') first."

    try:
        df = pd.read_csv(path)
        if operation == "describe":
            return f"DataFrame: {len(df)} rows × {len(df.columns)} columns\n\n{df.describe().to_string()}"
        elif operation == "head":
            return f"First 5 rows:\n{df.head().to_string()}"
        elif operation == "value_counts":
            # Value counts du premier column catégoriel
            cat_cols = df.select_dtypes(include=["object"]).columns
            if len(cat_cols) > 0:
                return f"Value counts for '{cat_cols[0]}':\n{df[cat_cols[0]].value_counts().to_string()}"
            return f"No categorical columns. Columns: {list(df.columns)}"
        elif operation == "correlation":
            num_df = df.select_dtypes(include=["number"])
            if len(num_df.columns) > 1:
                return f"Correlation matrix:\n{num_df.corr().to_string()}"
            return "Not enough numeric columns for correlation."
        else:
            return f"Unknown operation: '{operation}'. Use: describe, head, value_counts, correlation."
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def real_json_query(filepath: str, query: str) -> str:
    """Requête simple sur un fichier JSON."""
    path = _resolve_sandbox_path(filepath)
    if not path.exists():
        return f"ERROR: File not found: '{filepath}'"

    try:
        data = json.loads(path.read_text())

        # Support basique de JSONPath-like queries
        # $.key, $.key.subkey, $.array[0], $.array[*].field
        parts = query.replace("$.", "").replace("$", "").split(".")
        current = data
        for part in parts:
            if not part:
                continue
            # Array index
            idx_match = re.match(r'(\w+)\[(\d+|\*)\]', part)
            if idx_match:
                key, idx = idx_match.group(1), idx_match.group(2)
                if key:
                    current = current[key]
                if idx == "*":
                    if isinstance(current, list):
                        return json.dumps(current, indent=2, default=str)[:5000]
                else:
                    current = current[int(idx)]
            else:
                if isinstance(current, dict):
                    current = current[part]
                elif isinstance(current, list):
                    # Filter : chercher dans la liste
                    matches = [item for item in current
                              if isinstance(item, dict) and part in str(item.values())]
                    return f"Found {len(matches)} matches:\n{json.dumps(matches[:10], indent=2, default=str)}"

        return json.dumps(current, indent=2, default=str)[:5000]
    except (KeyError, IndexError, TypeError) as e:
        return f"Query '{query}' failed: {type(e).__name__}: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def real_create_plot(plot_type: str, data: str, title: str = "Plot") -> str:
    """Crée un vrai plot avec matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "ERROR: matplotlib not installed. Call install_package('matplotlib') first."

    try:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Parser les données (format flexible)
        parsed = _parse_plot_data(data)

        if plot_type == "bar":
            ax.bar(parsed["labels"], parsed["values"])
        elif plot_type == "line":
            ax.plot(parsed["values"])
        elif plot_type == "scatter":
            if len(parsed.get("values2", [])) == len(parsed["values"]):
                ax.scatter(parsed["values"], parsed["values2"])
            else:
                ax.scatter(range(len(parsed["values"])), parsed["values"])
        elif plot_type == "histogram":
            ax.hist(parsed["values"], bins=min(20, len(parsed["values"])))
        else:
            return f"ERROR: Unknown plot type '{plot_type}'. Use: bar, line, scatter, histogram."

        ax.set_title(title)
        plot_path = SANDBOX_DIR / f"plot_{uuid.uuid4().hex[:8]}.png"
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return f"Plot saved to {plot_path}. Type: {plot_type}, Title: '{title}', Data points: {len(parsed['values'])}."
    except Exception as e:
        return f"ERROR creating plot: {type(e).__name__}: {e}"


def real_unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Conversion d'unités avec table de conversion réelle."""
    conversions = {
        # Longueur
        ("km", "miles"): 0.621371, ("miles", "km"): 1.60934,
        ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
        ("cm", "inches"): 0.393701, ("inches", "cm"): 2.54,
        ("km", "m"): 1000, ("m", "km"): 0.001,
        # Poids
        ("kg", "lbs"): 2.20462, ("lbs", "kg"): 0.453592,
        ("g", "oz"): 0.035274, ("oz", "g"): 28.3495,
        # Volume
        ("l", "gallons"): 0.264172, ("gallons", "l"): 3.78541,
        ("ml", "fl_oz"): 0.033814, ("fl_oz", "ml"): 29.5735,
        # Temps
        ("hours", "minutes"): 60, ("minutes", "hours"): 1/60,
        ("days", "hours"): 24, ("hours", "days"): 1/24,
        # Données
        ("gb", "mb"): 1024, ("mb", "gb"): 1/1024,
        ("tb", "gb"): 1024, ("gb", "tb"): 1/1024,
    }

    f, t = from_unit.lower().strip(), to_unit.lower().strip()

    # Température (fonctions, pas facteurs)
    if f in ("celsius", "c", "°c") and t in ("fahrenheit", "f", "°f"):
        result = value * 9 / 5 + 32
        return f"{value}°C = {result:.2f}°F"
    if f in ("fahrenheit", "f", "°f") and t in ("celsius", "c", "°c"):
        result = (value - 32) * 5 / 9
        return f"{value}°F = {result:.2f}°C"
    if f in ("celsius", "c", "°c") and t in ("kelvin", "k"):
        result = value + 273.15
        return f"{value}°C = {result:.2f}K"
    if f in ("kelvin", "k") and t in ("celsius", "c", "°c"):
        result = value - 273.15
        return f"{value}K = {result:.2f}°C"

    key = (f, t)
    if key in conversions:
        result = value * conversions[key]
        return f"{value} {from_unit} = {result:.6g} {to_unit}"

    return f"ERROR: Unknown conversion: {from_unit} → {to_unit}. Supported units: {sorted(set(k[0] for k in conversions))}"


def real_get_current_datetime(timezone_str: str = "UTC") -> str:
    """Retourne la date/heure réelle."""
    try:
        now_utc = datetime.now(timezone.utc)

        # Mapping des timezones courants
        tz_offsets = {
            "utc": 0, "gmt": 0,
            "est": -5, "cst": -6, "mst": -7, "pst": -8,
            "cet": 1, "eet": 2,
            "jst": 9, "kst": 9, "ist": 5.5,
            "europe/paris": 1, "europe/london": 0,
            "america/new_york": -5, "america/chicago": -6,
            "america/los_angeles": -8,
            "asia/tokyo": 9, "asia/shanghai": 8,
        }

        tz_key = timezone_str.lower().strip()
        offset = tz_offsets.get(tz_key, None)

        if offset is not None:
            target_time = now_utc + timedelta(hours=offset)
            day_name = target_time.strftime("%A")
            return (
                f"Current date/time ({timezone_str}): "
                f"{target_time.strftime('%Y-%m-%dT%H:%M:%S')}"
                f"{'+' if offset >= 0 else ''}{int(offset):02d}:{'30' if offset % 1 else '00'} "
                f"({day_name})"
            )
        else:
            # Fallback UTC
            day_name = now_utc.strftime("%A")
            return (
                f"Current date/time (UTC, requested '{timezone_str}' not recognized): "
                f"{now_utc.strftime('%Y-%m-%dT%H:%M:%S+00:00')} ({day_name})"
            )
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def real_generate_id() -> str:
    """Génère un vrai UUID."""
    return f"Generated ID: {uuid.uuid4()}"


def real_summarize_text(text: str, max_length: int = 100) -> str:
    """
    Résumé par extraction (pas d'appel LLM — on ne veut pas de récursion).
    Extrait les premières phrases jusqu'à max_length mots.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    summary_words = []
    for sentence in sentences:
        words = sentence.split()
        if len(summary_words) + len(words) <= max_length:
            summary_words.extend(words)
        else:
            break

    if not summary_words and sentences:
        # Au moins la première phrase
        summary_words = sentences[0].split()[:max_length]

    summary = " ".join(summary_words)
    original_words = len(text.split())
    return (
        f"Summary ({len(summary_words)} words, from {original_words} original):\n"
        f"{summary}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# API EXECUTORS — appels à des APIs gratuites
# ═══════════════════════════════════════════════════════════════════════════

def api_web_search(query: str, num_results: int = 3) -> str:
    """
    Recherche web via DuckDuckGo (pas de clé API nécessaire).
    Fallback : recherche simulée si l'API est indisponible.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        if not results:
            return f"No results found for '{query}'."
        lines = []
        for i, r in enumerate(results):
            title = r.get("title", "No title")
            body = r.get("body", "")[:200]
            href = r.get("href", "")
            lines.append(f"[{i+1}] {title}\n    {body}\n    URL: {href}")
        return "\n\n".join(lines)
    except ImportError:
        logger.warning("duckduckgo_search not installed — using fallback")
        return _fallback_web_search(query, num_results)
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e} — using fallback")
        return _fallback_web_search(query, num_results)


def _fallback_web_search(query: str, num_results: int = 3) -> str:
    """Fallback : recherche via requests + scraping minimal."""
    try:
        import requests
        # Utiliser l'API Lite de DuckDuckGo
        resp = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; JEPA-Agent/0.1)"},
            timeout=10,
        )
        if resp.status_code == 200:
            # Parser les résultats (format simplifié)
            text = resp.text
            # Extraire les liens et titres basiques
            links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*class="result-link"[^>]*>([^<]+)</a>', text)
            if links:
                lines = [f"[{i+1}] {title.strip()}\n    URL: {url}" for i, (url, title) in enumerate(links[:num_results])]
                return "\n\n".join(lines)
        return f"Web search for '{query}': API unavailable. Please install duckduckgo_search: pip install duckduckgo_search"
    except Exception:
        return f"Web search for '{query}': Network unavailable. Install: pip install duckduckgo_search"


def api_fetch_url(url: str) -> str:
    """Fetch le contenu d'une URL réelle."""
    try:
        import requests
    except ImportError:
        return "ERROR: requests not installed. Call install_package('requests') first."

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JEPA-Agent/0.1)"},
            timeout=15,
            allow_redirects=True,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        if "json" in content_type:
            return json.dumps(resp.json(), indent=2)[:10_000]

        if "html" in content_type:
            # Extraire le texte du HTML (basique)
            text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 10_000:
                text = text[:10_000] + "\n\n[... truncated]"
            return f"Content from {url}:\n{text}"

        # Texte brut
        text = resp.text[:10_000]
        return f"Content from {url}:\n{text}"

    except Exception as e:
        return f"ERROR fetching {url}: {type(e).__name__}: {e}"


def api_wikipedia_lookup(topic: str) -> str:
    """Recherche Wikipedia via l'API REST officielle (gratuite, pas de clé)."""
    try:
        import requests
    except ImportError:
        return "ERROR: requests not installed."

    try:
        # API Wikipedia summary
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "JEPA-Agent/0.1"})

        if resp.status_code == 404:
            # Essayer la recherche
            search_url = "https://en.wikipedia.org/w/api.php"
            search_resp = requests.get(search_url, params={
                "action": "opensearch", "search": topic, "limit": 3, "format": "json"
            }, timeout=10)
            if search_resp.status_code == 200:
                data = search_resp.json()
                if len(data) > 1 and data[1]:
                    suggestions = ", ".join(data[1][:3])
                    return f"No Wikipedia article found for '{topic}'. Did you mean: {suggestions}?"
            return f"No Wikipedia article found for '{topic}'."

        resp.raise_for_status()
        data = resp.json()

        title = data.get("title", topic)
        extract = data.get("extract", "No extract available.")
        description = data.get("description", "")
        url_page = data.get("content_urls", {}).get("desktop", {}).get("page", "")

        result = f"Wikipedia: {title}"
        if description:
            result += f"\n({description})"
        result += f"\n\n{extract}"
        if url_page:
            result += f"\n\nSource: {url_page}"

        return result

    except Exception as e:
        return f"ERROR: Wikipedia lookup failed: {type(e).__name__}: {e}"


def api_arxiv_search(query: str, max_results: int = 3) -> str:
    """Recherche arXiv via l'API Atom (gratuite, pas de clé)."""
    try:
        import requests
    except ImportError:
        return "ERROR: requests not installed."

    try:
        url = "http://export.arxiv.org/api/query"
        resp = requests.get(url, params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }, timeout=15)
        resp.raise_for_status()

        # Parser le XML Atom (basique, sans dépendance lxml)
        entries = re.findall(r'<entry>(.*?)</entry>', resp.text, re.DOTALL)
        if not entries:
            return f"No arXiv results found for '{query}'."

        lines = []
        for i, entry in enumerate(entries):
            title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            title = title.group(1).strip().replace("\n", " ") if title else "No title"

            arxiv_id = re.search(r'<id>(.*?)</id>', entry)
            arxiv_id = arxiv_id.group(1).strip() if arxiv_id else ""
            # Extraire l'ID court
            short_id = arxiv_id.split("/abs/")[-1] if "/abs/" in arxiv_id else arxiv_id

            published = re.search(r'<published>(.*?)</published>', entry)
            published = published.group(1)[:10] if published else ""

            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            summary = summary.group(1).strip().replace("\n", " ")[:200] if summary else ""

            authors = re.findall(r'<name>(.*?)</name>', entry)
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += f" et al. ({len(authors)} authors)"

            lines.append(
                f"[{i+1}] {title}\n"
                f"    Authors: {author_str}\n"
                f"    Published: {published}\n"
                f"    ID: {short_id}\n"
                f"    {summary}..."
            )

        return "\n\n".join(lines)

    except Exception as e:
        return f"ERROR: arXiv search failed: {type(e).__name__}: {e}"


def api_get_weather(location: str) -> str:
    """
    Météo via OpenWeatherMap (clé gratuite) ou wttr.in (sans clé).
    """
    # Essayer wttr.in d'abord (pas de clé nécessaire)
    try:
        import requests
        resp = requests.get(
            f"https://wttr.in/{location}?format=j1",
            timeout=30,
            headers={"User-Agent": "JEPA-Agent/0.1"},
        )
        if resp.status_code == 200:
            data = resp.json()
            current = data.get("current_condition", [{}])[0]
            temp_c = current.get("temp_C", "?")
            feels_like = current.get("FeelsLikeC", "?")
            humidity = current.get("humidity", "?")
            desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
            wind_kmph = current.get("windspeedKmph", "?")
            wind_dir = current.get("winddir16Point", "")

            area = data.get("nearest_area", [{}])[0]
            area_name = area.get("areaName", [{}])[0].get("value", location)
            country = area.get("country", [{}])[0].get("value", "")

            return (
                f"Weather in {area_name}, {country}:\n"
                f"  Condition: {desc}\n"
                f"  Temperature: {temp_c}°C (feels like {feels_like}°C)\n"
                f"  Humidity: {humidity}%\n"
                f"  Wind: {wind_kmph} km/h {wind_dir}"
            )
    except ImportError:
        return "ERROR: requests not installed."
    except Exception as e:
        logger.warning(f"wttr.in failed: {e} — using mock fallback")
        return mock_get_weather(location)

    # Fallback OpenWeatherMap
    if OPENWEATHER_API_KEY:
        try:
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": location, "appid": OPENWEATHER_API_KEY, "units": "metric"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                temp = data["main"]["temp"]
                desc = data["weather"][0]["description"]
                humidity = data["main"]["humidity"]
                return f"Weather in {location}: {temp}°C, {desc}. Humidity: {humidity}%."
        except Exception:
            pass

    return f"ERROR: Weather data unavailable for '{location}'. Check network connectivity."


# ═══════════════════════════════════════════════════════════════════════════
# MOCK EXECUTORS (réalistes) — pour les tools à effets de bord
# ═══════════════════════════════════════════════════════════════════════════
# Ces mocks retournent des formats identiques aux vraies APIs
# pour que les hidden states de l'agent soient réalistes.

def mock_send_email(to: str, subject: str, body: str) -> str:
    """Mock réaliste — format identique à l'API Gmail/SendGrid."""
    msg_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()
    return json.dumps({
        "status": "sent",
        "message_id": f"<{msg_id}@jepa-agent.local>",
        "to": to,
        "subject": subject,
        "body_length": len(body),
        "timestamp": timestamp,
        "headers": {
            "From": "agent@jepa-agent.local",
            "To": to,
            "Subject": subject,
            "Date": timestamp,
            "Message-ID": f"<{msg_id}@jepa-agent.local>",
            "Content-Type": "text/plain; charset=utf-8",
        },
    }, indent=2)


def mock_send_slack_message(channel: str, message: str) -> str:
    """Mock réaliste — format identique à l'API Slack."""
    ts = str(time.time())
    return json.dumps({
        "ok": True,
        "channel": channel,
        "ts": ts,
        "message": {
            "type": "message",
            "subtype": "bot_message",
            "text": message,
            "ts": ts,
            "username": "jepa-agent",
            "bot_id": "B0JEPA01",
        },
    }, indent=2)


def mock_schedule_meeting(title: str, date: str, duration_minutes: int = 30,
                          attendees: str = "") -> str:
    """Mock réaliste — format identique à l'API Google Calendar."""
    event_id = uuid.uuid4().hex[:16]
    attendee_list = [a.strip() for a in attendees.split(",") if a.strip()] if attendees else []
    return json.dumps({
        "status": "confirmed",
        "kind": "calendar#event",
        "id": event_id,
        "summary": title,
        "start": {"dateTime": date, "timeZone": "UTC"},
        "end": {"dateTime": f"{date} +{duration_minutes}min", "timeZone": "UTC"},
        "duration_minutes": duration_minutes,
        "attendees": [
            {"email": a, "responseStatus": "needsAction"}
            for a in attendee_list
        ],
        "htmlLink": f"https://calendar.google.com/event?eid={event_id}",
        "created": datetime.now(timezone.utc).isoformat(),
    }, indent=2)


def mock_create_todo(task: str, priority: str = "medium", due_date: str = "") -> str:
    """Mock réaliste — format type Todoist/Linear API."""
    task_id = uuid.uuid4().hex[:8]
    return json.dumps({
        "id": task_id,
        "content": task,
        "priority": {"low": 1, "medium": 2, "high": 3}.get(priority, 2),
        "priority_label": priority,
        "due": {"date": due_date} if due_date else None,
        "is_completed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "url": f"https://app.todoist.com/task/{task_id}",
    }, indent=2)


def mock_translate_text(text: str, target_language: str,
                        source_language: str = "auto") -> str:
    """
    Mock réaliste — format type Google Translate API.
    Note : pour une vraie traduction, on pourrait utiliser
    l'API gratuite de LibreTranslate ou deep-translator.
    """
    # Tenter une vraie traduction avec deep-translator
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(
            source=source_language if source_language != "auto" else "auto",
            target=target_language
        ).translate(text)
        return json.dumps({
            "translated_text": translated,
            "source_language": source_language,
            "target_language": target_language,
            "confidence": 0.98,
            "engine": "google_translate",
        }, indent=2)
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback : mock avec format réaliste
    return json.dumps({
        "translated_text": f"[{target_language} translation of: {text[:100]}]",
        "source_language": source_language,
        "target_language": target_language,
        "confidence": 0.0,
        "engine": "mock (install deep-translator for real translation)",
        "note": "pip install deep-translator for actual translations",
    }, indent=2)


def mock_get_weather(location: str) -> str:
    """
    Mock réaliste pour la météo — retourne un texte formaté utilisable par les tests.
    """
    # Valeurs fixes pour garantir des tests déterministes
    data = {
        "location": location,
        "condition": "Partly cloudy",
        "temperature_c": 18,
        "feels_like_c": 17,
        "humidity": 56,
        "wind_kmph": 10,
        "wind_dir": "NW",
    }

    return (
        f"Mock weather in {data['location']}:\n"
        f"  Condition: {data['condition']}\n"
        f"  Temperature: {data['temperature_c']}°C (feels like {data['feels_like_c']}°C)\n"
        f"  Humidity: {data['humidity']}%\n"
        f"  Wind: {data['wind_kmph']} km/h {data['wind_dir']}\n"
        f"  Source: mock_get_weather"
    )


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_sandbox_path(filepath: str) -> Path:
    """Résout un chemin relatif au sandbox, empêche le path traversal."""
    path = Path(filepath)
    if path.is_absolute():
        # Vérifier que c'est dans le sandbox
        resolved = path.resolve()
        sandbox_resolved = SANDBOX_DIR.resolve()
        if not str(resolved).startswith(str(sandbox_resolved)):
            # Rediriger dans le sandbox
            path = SANDBOX_DIR / path.name
        else:
            return resolved
    else:
        path = SANDBOX_DIR / path

    return path.resolve()


def _parse_plot_data(data: str) -> dict:
    """Parse des données pour les plots (format flexible)."""
    result = {"labels": [], "values": [], "values2": []}

    # Format "key=value, key=value"
    kv_matches = re.findall(r'(\w[\w\s]*?)\s*[=:]\s*([\d.]+)', data)
    if kv_matches:
        result["labels"] = [k.strip() for k, v in kv_matches]
        result["values"] = [float(v) for k, v in kv_matches]
        return result

    # Format liste de nombres
    numbers = re.findall(r'[\d.]+', data)
    if numbers:
        result["values"] = [float(n) for n in numbers]
        result["labels"] = [str(i) for i in range(len(result["values"]))]
        return result

    # Fallback
    result["values"] = [1, 2, 3]
    result["labels"] = ["a", "b", "c"]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTOR MAP — connecte chaque tool à son exécuteur
# ═══════════════════════════════════════════════════════════════════════════

EXECUTOR_MAP: dict[str, callable] = {
    # REAL — exécution authentique
    "python_execute": real_python_execute,
    "python_eval": real_python_eval,
    "calculator": real_calculator,
    "shell_command": real_shell_command,
    "install_package": real_install_package,
    "read_file": real_read_file,
    "write_file": real_write_file,
    "list_directory": real_list_directory,
    "csv_analyze": real_csv_analyze,
    "json_query": real_json_query,
    "create_plot": real_create_plot,
    "unit_convert": real_unit_convert,
    "get_current_datetime": real_get_current_datetime,
    "generate_id": real_generate_id,
    "summarize_text": real_summarize_text,

    # API — appels réseau réels
    "web_search": api_web_search,
    "fetch_url": api_fetch_url,
    "wikipedia_lookup": api_wikipedia_lookup,
    "arxiv_search": api_arxiv_search,
    "get_weather": api_get_weather,

    # MOCK réaliste — effets de bord simulés avec format API authentique
    "send_email": mock_send_email,
    "send_slack_message": mock_send_slack_message,
    "schedule_meeting": mock_schedule_meeting,
    "create_todo": mock_create_todo,
    "translate_text": mock_translate_text,
}


def get_executor_type(tool_name: str) -> str:
    """Retourne le type d'exécuteur pour un tool donné."""
    real_tools = {
        "python_execute", "python_eval", "calculator", "shell_command",
        "install_package", "read_file", "write_file", "list_directory",
        "csv_analyze", "json_query", "create_plot", "unit_convert",
        "get_current_datetime", "generate_id", "summarize_text",
    }
    api_tools = {
        "web_search", "fetch_url", "wikipedia_lookup", "arxiv_search", "get_weather",
    }
    if tool_name in real_tools:
        return "real"
    if tool_name in api_tools:
        return "api"
    return "mock"
