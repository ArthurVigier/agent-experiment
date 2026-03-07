"""
tools.py
Définitions des tools pour l'agent baseline Sprint 0.

25 tools répartis en 5 clusters fonctionnels.
Chaque tool a un schema JSON (compatible OpenAI function calling format),
un mock executor pour les tests, et des métadonnées pour l'analyse.

Convention : docstrings français, code anglais.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import json
import random
import math


@dataclass
class Tool:
    """Définition complète d'un tool."""
    name: str
    description: str
    category: str  # search, code, file, communication, data
    parameters: dict[str, Any]  # JSON Schema des paramètres
    executor: Callable[..., str]  # Fonction qui simule l'exécution
    reversible: bool = True  # Le tool peut-il être annulé ?
    deterministic: bool = True  # Même input → même output ?

    def to_schema(self) -> dict:
        """Format compatible OpenAI function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs) -> str:
        """Exécute le tool avec les paramètres donnés."""
        try:
            return self.executor(**kwargs)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"


# ─── Mock Executors ─────────────────────────────────────────────────────────
# Retournent des résultats réalistes mais synthétiques.
# Suffisant pour générer des traces et tester le pipeline.

def _web_search(query: str, num_results: int = 3) -> str:
    results = [
        f"[{i+1}] Result for '{query}': Lorem ipsum finding #{random.randint(100,999)} "
        f"— source: example{random.randint(1,50)}.com"
        for i in range(num_results)
    ]
    return "\n".join(results)


def _fetch_url(url: str) -> str:
    return (
        f"Content from {url}:\n"
        f"Title: Sample Page About {url.split('/')[-1]}\n"
        f"Body: This page contains information about the requested topic. "
        f"Key findings include points A, B, and C. Last updated 2025-01-15."
    )


def _wikipedia_lookup(topic: str) -> str:
    return (
        f"Wikipedia: {topic}\n"
        f"{topic} is a subject of significant interest. "
        f"It was first described in {random.randint(1900, 2020)}. "
        f"Key aspects include its historical development and modern applications."
    )


def _arxiv_search(query: str, max_results: int = 3) -> str:
    results = [
        f"[{i+1}] arXiv:{random.randint(2300,2512)}.{random.randint(10000,99999)} "
        f"- '{query} in {['Neural Networks', 'Transformers', 'LLMs', 'RL'][i % 4]}' "
        f"(2025, {random.randint(5,50)} citations)"
        for i in range(max_results)
    ]
    return "\n".join(results)


def _get_weather(location: str) -> str:
    temp = random.randint(-5, 35)
    conditions = random.choice(["sunny", "cloudy", "rainy", "snowy", "partly cloudy"])
    return f"Weather in {location}: {temp}°C, {conditions}. Humidity: {random.randint(30,90)}%."


def _python_execute(code: str) -> str:
    # Simulation sécurisée — on ne fait PAS de vrai exec
    if "print" in code:
        if "sum(range" in code:
            # Cas fréquent dans les benchmarks
            try:
                n = int(code.split("range(")[1].split(")")[0])
                return str(n * (n - 1) // 2)
            except (IndexError, ValueError):
                return "499500"  # sum(range(1000))
        return f"[stdout]: simulated output for: {code[:80]}"
    if "def " in code:
        return "Function defined successfully."
    if "import" in code:
        return "Module imported successfully."
    return f"Code executed. No output. Variables defined."


def _python_eval(expression: str) -> str:
    # Évaluation safe d'expressions mathématiques simples
    try:
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "int": int, "float": float,
            "pow": pow, "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
            "cos": math.cos, "sin": math.sin, "log": math.log,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception:
        return f"Evaluated: {expression} → [simulated result]"


def _shell_command(command: str) -> str:
    if "ls" in command:
        return "file1.py  file2.csv  README.md  data/  results/"
    if "cat" in command:
        return "File contents: [simulated file output]"
    if "pip" in command:
        return "Successfully installed package."
    if "git" in command:
        return "Git operation completed."
    return f"$ {command}\n[simulated output]"


def _install_package(package_name: str) -> str:
    return f"Successfully installed {package_name} (version {random.randint(1,5)}.{random.randint(0,20)}.{random.randint(0,9)})"


def _create_plot(plot_type: str, data: str, title: str = "Plot") -> str:
    return f"Created {plot_type} chart titled '{title}' with data: {data[:100]}. Saved to /tmp/plot.png."


def _read_file(filepath: str) -> str:
    ext = filepath.rsplit(".", 1)[-1] if "." in filepath else "txt"
    if ext == "csv":
        return "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n... (100 rows total)"
    if ext == "json":
        return '{"key": "value", "items": [1, 2, 3], "nested": {"a": true}}'
    if ext == "py":
        return "# Python file\ndef main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()"
    return f"Content of {filepath}: [simulated text content, 250 words]"


def _write_file(filepath: str, content: str) -> str:
    return f"Successfully wrote {len(content)} characters to {filepath}."


def _list_directory(path: str = ".") -> str:
    return f"Contents of {path}/:\n  data.csv (12KB)\n  model.py (8KB)\n  config.yaml (1KB)\n  results/ (directory)\n  README.md (3KB)"


def _csv_analyze(filepath: str, operation: str = "describe") -> str:
    if operation == "describe":
        return "DataFrame: 100 rows × 5 columns\nColumn stats:\n  accuracy: mean=0.847, std=0.032\n  loss: mean=0.423, std=0.089\n  epoch: mean=5.0, std=2.87"
    if operation == "head":
        return "   accuracy  loss  epoch  lr     model\n0  0.823     0.45  1      0.001  baseline\n1  0.856     0.39  2      0.001  baseline"
    return f"CSV analysis ({operation}): [simulated result]"


def _json_query(filepath: str, query: str) -> str:
    return f"Query '{query}' on {filepath}: Found 3 matching entries. First: {{'id': 1, 'value': 'result'}}"


def _send_email(to: str, subject: str, body: str) -> str:
    return f"Email sent to {to} with subject '{subject}'. Body length: {len(body)} chars."


def _send_slack_message(channel: str, message: str) -> str:
    return f"Message posted to #{channel}: '{message[:50]}...'" if len(message) > 50 else f"Message posted to #{channel}: '{message}'"


def _schedule_meeting(title: str, date: str, duration_minutes: int = 30, attendees: str = "") -> str:
    return f"Meeting '{title}' scheduled for {date}, {duration_minutes}min. Attendees: {attendees or 'none specified'}."


def _create_todo(task: str, priority: str = "medium", due_date: str = "") -> str:
    return f"TODO created: '{task}' [priority: {priority}]{f', due: {due_date}' if due_date else ''}."


def _translate_text(text: str, target_language: str, source_language: str = "auto") -> str:
    return f"Translation ({source_language} → {target_language}): [simulated translation of '{text[:50]}']"


def _summarize_text(text: str, max_length: int = 100) -> str:
    words = text.split()
    if len(words) <= max_length:
        return f"Summary: {text}"
    return f"Summary ({max_length} words): {' '.join(words[:max_length])}..."


def _calculator(expression: str) -> str:
    return _python_eval(expression)


def _unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    conversions = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        conv = conversions[key]
        result = conv(value) if callable(conv) else value * conv
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"{value} {from_unit} ≈ {value * random.uniform(0.5, 2.0):.4f} {to_unit} (approximate)"


def _get_current_datetime(timezone: str = "UTC") -> str:
    return f"Current date/time ({timezone}): 2026-03-04T14:30:00+00:00 (Wednesday)"


# ─── Tool Registry ──────────────────────────────────────────────────────────

TOOLS: list[Tool] = [
    # ── Search (5 tools) ──
    Tool(
        name="web_search",
        description="Search the web for current information on any topic.",
        category="search",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results", "default": 3},
            },
            "required": ["query"],
        },
        executor=_web_search,
        deterministic=False,
    ),
    Tool(
        name="fetch_url",
        description="Fetch and read the content of a web page at a given URL.",
        category="search",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            },
            "required": ["url"],
        },
        executor=_fetch_url,
        deterministic=False,
    ),
    Tool(
        name="wikipedia_lookup",
        description="Look up a topic on Wikipedia and return a summary.",
        category="search",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to look up"},
            },
            "required": ["topic"],
        },
        executor=_wikipedia_lookup,
        deterministic=False,
    ),
    Tool(
        name="arxiv_search",
        description="Search for academic papers on arXiv by topic or keywords.",
        category="search",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results", "default": 3},
            },
            "required": ["query"],
        },
        executor=_arxiv_search,
        deterministic=False,
    ),
    Tool(
        name="get_weather",
        description="Get the current weather for a location.",
        category="search",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or location name"},
            },
            "required": ["location"],
        },
        executor=_get_weather,
        deterministic=False,
    ),

    # ── Code (5 tools) ──
    Tool(
        name="python_execute",
        description="Execute Python code and return the output.",
        category="code",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
        executor=_python_execute,
        reversible=False,
    ),
    Tool(
        name="python_eval",
        description="Evaluate a Python expression and return the result.",
        category="code",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Python expression to evaluate"},
            },
            "required": ["expression"],
        },
        executor=_python_eval,
    ),
    Tool(
        name="shell_command",
        description="Execute a shell command and return the output.",
        category="code",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        },
        executor=_shell_command,
        reversible=False,
    ),
    Tool(
        name="install_package",
        description="Install a Python package using pip.",
        category="code",
        parameters={
            "type": "object",
            "properties": {
                "package_name": {"type": "string", "description": "Package name to install"},
            },
            "required": ["package_name"],
        },
        executor=_install_package,
        reversible=False,
    ),
    Tool(
        name="create_plot",
        description="Create a data visualization plot (bar, line, scatter, histogram).",
        category="code",
        parameters={
            "type": "object",
            "properties": {
                "plot_type": {"type": "string", "enum": ["bar", "line", "scatter", "histogram"]},
                "data": {"type": "string", "description": "Data description or values"},
                "title": {"type": "string", "description": "Plot title", "default": "Plot"},
            },
            "required": ["plot_type", "data"],
        },
        executor=_create_plot,
    ),

    # ── File (5 tools) ──
    Tool(
        name="read_file",
        description="Read the contents of a file (text, CSV, JSON, Python, etc.).",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the file"},
            },
            "required": ["filepath"],
        },
        executor=_read_file,
    ),
    Tool(
        name="write_file",
        description="Write content to a file. Creates the file if it doesn't exist.",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["filepath", "content"],
        },
        executor=_write_file,
        reversible=False,
    ),
    Tool(
        name="list_directory",
        description="List files and subdirectories in a directory.",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path", "default": "."},
            },
            "required": [],
        },
        executor=_list_directory,
    ),
    Tool(
        name="csv_analyze",
        description="Analyze a CSV file: describe stats, show head, compute aggregates.",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the CSV file"},
                "operation": {"type": "string", "enum": ["describe", "head", "value_counts", "correlation"], "default": "describe"},
            },
            "required": ["filepath"],
        },
        executor=_csv_analyze,
    ),
    Tool(
        name="json_query",
        description="Query a JSON file using a path expression or filter.",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the JSON file"},
                "query": {"type": "string", "description": "Query expression (e.g., '$.items[0].name')"},
            },
            "required": ["filepath", "query"],
        },
        executor=_json_query,
    ),

    # ── Communication (5 tools) ──
    Tool(
        name="send_email",
        description="Send an email to a recipient.",
        category="communication",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"},
            },
            "required": ["to", "subject", "body"],
        },
        executor=_send_email,
        reversible=False,
    ),
    Tool(
        name="send_slack_message",
        description="Post a message to a Slack channel.",
        category="communication",
        parameters={
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Slack channel name (without #)"},
                "message": {"type": "string", "description": "Message text"},
            },
            "required": ["channel", "message"],
        },
        executor=_send_slack_message,
        reversible=False,
    ),
    Tool(
        name="schedule_meeting",
        description="Schedule a meeting or calendar event.",
        category="communication",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Meeting title"},
                "date": {"type": "string", "description": "Date and time (ISO format or natural language)"},
                "duration_minutes": {"type": "integer", "description": "Duration in minutes", "default": 30},
                "attendees": {"type": "string", "description": "Comma-separated list of attendees"},
            },
            "required": ["title", "date"],
        },
        executor=_schedule_meeting,
        reversible=True,
    ),
    Tool(
        name="create_todo",
        description="Create a to-do item or task.",
        category="communication",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task description"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
                "due_date": {"type": "string", "description": "Due date (optional)"},
            },
            "required": ["task"],
        },
        executor=_create_todo,
    ),
    Tool(
        name="translate_text",
        description="Translate text from one language to another.",
        category="communication",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "target_language": {"type": "string", "description": "Target language code (e.g., 'fr', 'en', 'de')"},
                "source_language": {"type": "string", "description": "Source language (auto-detect if omitted)", "default": "auto"},
            },
            "required": ["text", "target_language"],
        },
        executor=_translate_text,
    ),

    # ── Data / Utility (5 tools) ──
    Tool(
        name="calculator",
        description="Evaluate a mathematical expression.",
        category="data",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression (e.g., '2**10 + sqrt(144)')"},
            },
            "required": ["expression"],
        },
        executor=_calculator,
    ),
    Tool(
        name="unit_convert",
        description="Convert a value between units (length, weight, temperature).",
        category="data",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Value to convert"},
                "from_unit": {"type": "string", "description": "Source unit"},
                "to_unit": {"type": "string", "description": "Target unit"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
        executor=_unit_convert,
    ),
    Tool(
        name="summarize_text",
        description="Summarize a long text into a shorter version.",
        category="data",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
                "max_length": {"type": "integer", "description": "Maximum summary length in words", "default": 100},
            },
            "required": ["text"],
        },
        executor=_summarize_text,
    ),
    Tool(
        name="get_current_datetime",
        description="Get the current date and time in a given timezone.",
        category="data",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Timezone (e.g., 'UTC', 'Europe/Paris')", "default": "UTC"},
            },
            "required": [],
        },
        executor=_get_current_datetime,
    ),
    Tool(
        name="generate_id",
        description="Generate a unique identifier (UUID).",
        category="data",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        executor=lambda: f"Generated ID: {random.randint(10**11, 10**12-1):012d}",
    ),
]


# ─── Wire real executors ─────────────────────────────────────────────────────

def _wire_real_executors():
    """
    Remplace les mock executors par les vrais exécuteurs de executors.py.
    Appelé au moment de l'import pour que TOOL_MAP utilise les vrais backends.
    """
    try:
        from baselines.executors import EXECUTOR_MAP
        for tool in TOOLS:
            if tool.name in EXECUTOR_MAP:
                tool.executor = EXECUTOR_MAP[tool.name]
    except ImportError:
        pass  # executors.py pas disponible — garder les mocks intégrés

_wire_real_executors()


# ─── Helpers ────────────────────────────────────────────────────────────────

TOOL_MAP: dict[str, Tool] = {t.name: t for t in TOOLS}


def get_tools_by_category(category: str) -> list[Tool]:
    """Retourne les tools d'une catégorie donnée."""
    return [t for t in TOOLS if t.category == category]


def get_all_schemas() -> list[dict]:
    """Retourne tous les schemas au format OpenAI."""
    return [t.to_schema() for t in TOOLS]


def get_tool_names() -> list[str]:
    """Retourne la liste des noms de tools."""
    return [t.name for t in TOOLS]


def tool_description_block() -> str:
    """
    Génère le bloc de description des tools pour le system prompt.
    Format compact : nom — description.
    """
    lines = []
    for cat in ["search", "code", "file", "communication", "data"]:
        cat_tools = get_tools_by_category(cat)
        lines.append(f"\n## {cat.upper()} tools")
        for t in cat_tools:
            params = ", ".join(
                f"{k}: {v.get('type', '?')}" + (" (required)" if k in t.parameters.get("required", []) else "")
                for k, v in t.parameters.get("properties", {}).items()
            )
            lines.append(f"- **{t.name}**({params}): {t.description}")
    return "\n".join(lines)
