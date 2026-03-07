"""
tasks.py
Suite de tâches pour l'évaluation Sprint 0.

100 tâches réparties en 5 catégories × 4 complexités.
Chaque tâche a un ground truth : tools attendus et critère de succès.

Convention : docstrings français, code anglais.
"""

from dataclasses import dataclass


@dataclass
class Task:
    """Définition d'une tâche d'évaluation."""
    id: str
    prompt: str
    category: str  # search, code, file, communication, data, multi_step
    expected_tools: list[str]  # Tools qui devraient être appelés
    complexity: str  # single, chain, parallel, adversarial
    success_hint: str  # Description de ce qui constitue un succès


# ─── Single-tool tasks (40 tâches) ──────────────────────────────────────────

SINGLE_TASKS = [
    # Search
    Task("s01", "What is the current population of Tokyo?", "search",
         ["web_search"], "single", "Returns a plausible population number"),
    Task("s02", "Find me 3 recent papers about reinforcement learning from human feedback.",
         "search", ["arxiv_search"], "single", "Returns arxiv results about RLHF"),
    Task("s03", "What's the weather like in London right now?", "search",
         ["get_weather"], "single", "Must use get_weather tool and return valid weather data"),
    Task("s04", "Look up the Wikipedia article about the Turing Test.", "search",
         ["wikipedia_lookup"], "single", "Returns Wikipedia content about Turing Test"),
    Task("s05", "Search for the latest news about SpaceX Starship launches.", "search",
         ["web_search"], "single", "Returns recent news results"),
    Task("s06", "What is the weather forecast for New York City this weekend?", "search",
         ["get_weather"], "single", "Returns NYC weather"),
    Task("s07", "Find academic papers about transformer attention mechanisms.", "search",
         ["arxiv_search"], "single", "Returns relevant arxiv results"),
    Task("s08", "Look up what JEPA stands for on Wikipedia.", "search",
         ["wikipedia_lookup"], "single", "Returns JEPA info"),

    # Code
    Task("c01", "Calculate the result of 2^32 - 1.", "code",
         ["calculator"], "single", "Returns 4294967295"),
    Task("c02", "Run this Python code: print([x**2 for x in range(10)])", "code",
         ["python_execute"], "single", "Returns list of squares"),
    Task("c03", "What is the square root of 2 times pi?", "code",
         ["calculator"], "single", "Returns ~4.443"),
    Task("c04", "Evaluate the expression: sum(1/n for n in range(1, 101))", "code",
         ["python_eval"], "single", "Returns harmonic number ~5.187"),
    Task("c05", "Install the numpy package for me.", "code",
         ["install_package"], "single", "Confirms installation"),
    Task("c06", "Run: print('Hello' + ' ' + 'World')", "code",
         ["python_execute"], "single", "Returns Hello World"),
    Task("c07", "Create a bar chart showing sales by quarter: Q1=100, Q2=150, Q3=130, Q4=200.", "code",
         ["create_plot"], "single", "Creates a bar chart"),
    Task("c08", "Convert 100 degrees Fahrenheit to Celsius.", "code",
         ["unit_convert"], "single", "Returns ~37.78°C"),

    # File
    Task("f01", "Read the file 'data/report.txt' and tell me what's in it.", "file",
         ["read_file"], "single", "Returns file contents"),
    Task("f02", "What files are in the current directory?", "file",
         ["list_directory"], "single", "Returns directory listing"),
    Task("f03", "Show me the first few rows of the CSV file 'results.csv'.", "file",
         ["csv_analyze"], "single", "Returns head of CSV"),
    Task("f04", "Write 'Hello World' to a file called 'output.txt'.", "file",
         ["write_file"], "single", "Confirms file written"),
    Task("f05", "Read the JSON file 'config.json'.", "file",
         ["read_file"], "single", "Returns JSON contents"),
    Task("f06", "What are the statistics of the 'accuracy' column in 'metrics.csv'?", "file",
         ["csv_analyze"], "single", "Returns descriptive stats"),
    Task("f07", "Query the JSON file 'users.json' for all entries with role='admin'.", "file",
         ["json_query"], "single", "Returns matching entries"),
    Task("f08", "List all files in the 'data' directory.", "file",
         ["list_directory"], "single", "Returns directory listing"),

    # Communication
    Task("m01", "Send an email to alice@example.com saying the report is ready.", "communication",
         ["send_email"], "single", "Email sent confirmation"),
    Task("m02", "Post 'Deployment complete' to the #engineering Slack channel.", "communication",
         ["send_slack_message"], "single", "Message posted confirmation"),
    Task("m03", "Schedule a meeting with Bob for Friday at 3pm about the project review.", "communication",
         ["schedule_meeting"], "single", "Meeting scheduled"),
    Task("m04", "Create a high-priority to-do: Review pull request #42.", "communication",
         ["create_todo"], "single", "Todo created"),
    Task("m05", "Translate 'Bonjour, comment allez-vous?' to English.", "communication",
         ["translate_text"], "single", "Returns English translation"),
    Task("m06", "Schedule a 1-hour team standup for Monday at 9am.", "communication",
         ["schedule_meeting"], "single", "Meeting scheduled"),
    Task("m07", "Send a Slack message to #general: 'Happy Friday everyone!'", "communication",
         ["send_slack_message"], "single", "Message posted"),
    Task("m08", "Create a todo: Update documentation, due next Wednesday, medium priority.", "communication",
         ["create_todo"], "single", "Todo created"),

    # Data
    Task("d01", "What is the current date and time in UTC?", "data",
         ["get_current_datetime"], "single", "Returns current datetime"),
    Task("d02", "Convert 5 kilometers to miles.", "data",
         ["unit_convert"], "single", "Returns ~3.107 miles"),
    Task("d03", "Summarize this text: 'Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. These systems improve their performance on tasks without being explicitly programmed. Common approaches include supervised learning, unsupervised learning, and reinforcement learning.'", "data",
         ["summarize_text"], "single", "Returns a summary"),
    Task("d04", "Generate a unique ID for the new user account.", "data",
         ["generate_id"], "single", "Returns a UUID"),
    Task("d05", "How many pounds is 10 kilograms?", "data",
         ["unit_convert"], "single", "Returns ~22.05 lbs"),
    Task("d06", "What time is it in Europe/Paris?", "data",
         ["get_current_datetime"], "single", "Returns Paris time"),
    Task("d07", "Calculate: (17 * 23) + (sqrt(144) * 3)", "data",
         ["calculator"], "single", "Returns 427"),
    Task("d08", "Convert 72 degrees Fahrenheit to Celsius.", "data",
         ["unit_convert"], "single", "Returns ~22.22°C"),
]

# ─── Chain tasks (multi-step, séquentielles) ─────────────────────────────────

CHAIN_TASKS = [
    Task("ch01",
         "Search for the population of France, then calculate what 1% of that population is.",
         "multi_step", ["web_search", "calculator"], "chain",
         "Searches, then calculates 1% of the found number"),
    Task("ch02",
         "Read the file 'data.csv', look at the statistics, and write a summary to 'summary.txt'.",
         "multi_step", ["csv_analyze", "write_file"], "chain",
         "Reads CSV stats, then writes summary"),
    Task("ch03",
         "Find recent papers on JEPA, then send a Slack message to #research with the top 3 titles.",
         "multi_step", ["arxiv_search", "send_slack_message"], "chain",
         "Searches arxiv, then posts to Slack"),
    Task("ch04",
         "Check the weather in Tokyo, then send an email to travel@company.com with the forecast.",
         "multi_step", ["get_weather", "send_email"], "chain",
         "Gets weather, then emails it"),
    Task("ch05",
         "Look up the Wikipedia article on GPT-4, summarize it, and save the summary to 'gpt4_summary.txt'.",
         "multi_step", ["wikipedia_lookup", "summarize_text", "write_file"], "chain",
         "Wikipedia → summarize → write"),
    Task("ch06",
         "Calculate 2^64, then convert the result from bytes to gigabytes (divide by 1073741824).",
         "multi_step", ["calculator", "calculator"], "chain",
         "Two calculations in sequence"),
    Task("ch07",
         "Read 'report.txt', translate it to French, and save it as 'rapport.txt'.",
         "multi_step", ["read_file", "translate_text", "write_file"], "chain",
         "Read → translate → write"),
    Task("ch08",
         "Search for the latest Python version, then run code to check if our version matches.",
         "multi_step", ["web_search", "python_execute"], "chain",
         "Search → code execution"),
    Task("ch09",
         "List files in the current directory, read the first .csv file you find, and show its statistics.",
         "multi_step", ["list_directory", "csv_analyze"], "chain",
         "List → identify CSV → analyze"),
    Task("ch10",
         "Get the current time in UTC, then schedule a meeting for exactly 1 hour from now.",
         "multi_step", ["get_current_datetime", "schedule_meeting"], "chain",
         "Get time → schedule meeting"),
    Task("ch11",
         "Search for the GDP of Germany, convert it from EUR to USD (multiply by 1.08), and save the result to 'gdp.txt'.",
         "multi_step", ["web_search", "calculator", "write_file"], "chain",
         "Search → calculate → write"),
    Task("ch12",
         "Find papers on 'attention mechanism', read the first result's URL, and summarize what you find.",
         "multi_step", ["arxiv_search", "fetch_url", "summarize_text"], "chain",
         "Search → fetch → summarize"),
]

# ─── Adversarial / edge cases ────────────────────────────────────────────────

ADVERSARIAL_TASKS = [
    # Pas de tool nécessaire — l'agent devrait juste répondre
    Task("a01",
         "What is 2 + 2?",
         "data", [], "adversarial",
         "Answers 4 without calling any tool"),
    Task("a02",
         "Explain what a transformer architecture is in simple terms.",
         "data", [], "adversarial",
         "Explains without tools — this is general knowledge"),
    Task("a03",
         "Thank you for your help!",
         "data", [], "adversarial",
         "Responds politely without tool calls"),

    # Tool ambigu — plusieurs tools pourraient marcher
    Task("a04",
         "What is the value of pi to 10 decimal places?",
         "code", ["calculator", "python_eval"], "adversarial",
         "Either calculator or python_eval is fine"),
    Task("a05",
         "How far is it from Paris to London in miles?",
         "data", ["web_search", "unit_convert"], "adversarial",
         "Could search or just convert known distance"),

    # Requête impossible / tool inexistant
    Task("a06",
         "Generate an image of a sunset over the ocean.",
         "data", [], "adversarial",
         "Should explain no image generation tool is available"),
    Task("a07",
         "Deploy the application to production.",
         "code", [], "adversarial",
         "Should explain this isn't possible with available tools"),

    # Chaîne avec erreur recovery
    Task("a08",
         "Read the file 'nonexistent_file.xyz' and summarize it.",
         "file", ["read_file"], "adversarial",
         "Should handle the file not found error gracefully"),
]


# ─── Assemblage complet ─────────────────────────────────────────────────────

ALL_TASKS: list[Task] = SINGLE_TASKS + CHAIN_TASKS + ADVERSARIAL_TASKS

TASK_MAP: dict[str, Task] = {t.id: t for t in ALL_TASKS}


def get_tasks_by_category(category: str) -> list[Task]:
    """Retourne les tâches d'une catégorie."""
    return [t for t in ALL_TASKS if t.category == category]


def get_tasks_by_complexity(complexity: str) -> list[Task]:
    """Retourne les tâches d'une complexité donnée."""
    return [t for t in ALL_TASKS if t.complexity == complexity]


def task_stats() -> dict:
    """Statistiques sur la suite de tâches."""
    from collections import Counter
    cats = Counter(t.category for t in ALL_TASKS)
    comps = Counter(t.complexity for t in ALL_TASKS)
    return {
        "total": len(ALL_TASKS),
        "by_category": dict(cats),
        "by_complexity": dict(comps),
        "tools_covered": sorted(set(
            tool for t in ALL_TASKS for tool in t.expected_tools
        )),
    }
