"""
Central configuration for the CodeTrust data-collection pipeline.

All scripts (collect_human_code.py, generate_ai_code.py, dataset_builder.py)
import from here so language lists, paths, and cutoff dates stay consistent
across the whole pipeline.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Languages
# Python / Java / C++ per the original PRD, plus JavaScript so the project
# can also flag AI-generated frontend (React/vanilla JS) and backend
# (Node.js) code. Both are collected under one "javascript" label; see
# README "Splitting frontend vs backend JS" if you want that as a separate
# axis later (it would need a sub-label, not a new top-level language).
# ---------------------------------------------------------------------------
LANGUAGES = ["python", "java", "cpp", "javascript"]

EXTENSIONS = {
    "python": "py",
    "java": "java",
    "cpp": "cpp",
    "javascript": "js",
}

GITHUB_LANGUAGE_TAG = {
    "python": "Python",
    "java": "Java",
    "cpp": "C++",
    "javascript": "JavaScript",
}

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
HUMAN_DIR = DATA_DIR / "human_code"
AI_DIR = DATA_DIR / "ai_code"
SPLITS_DIR = DATA_DIR / "splits"

for lang in LANGUAGES:
    (HUMAN_DIR / lang).mkdir(parents=True, exist_ok=True)
    (AI_DIR / lang).mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

HUMAN_METADATA_PATH = DATA_DIR / "human_code_metadata.jsonl"
AI_METADATA_PATH = DATA_DIR / "ai_code_metadata.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

# ---------------------------------------------------------------------------
# Human-code collection (GitHub)
# ---------------------------------------------------------------------------
# Cutoff chosen to sit well before ChatGPT's public release (Nov 30, 2022)
# and before code-generation LLMs (Copilot GA, June 2022) were in wide use,
# to reduce the chance of unlabeled AI-assisted code contaminating the
# "human" class. Repos AND the specific file's last commit both must
# predate this date -- see collect_human_code.py.
HUMAN_CUTOFF_DATE = "2021-01-01"

# Permissive licenses only, so redistribution in a research dataset is safe.
ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "unlicense"}

MIN_REPO_STARS = 20
MAX_FILE_SIZE_BYTES = 50_000       # skip huge generated/vendored files
MIN_FILE_SIZE_BYTES = 200          # skip trivial stub files
SAMPLES_PER_LANGUAGE_TARGET = 500  # tune to your time/rate-limit budget

# Path fragments that usually indicate vendored/generated/test/bundled code
# we don't want in the human-authored set. JS repos especially need the
# bundle/minified/lockfile exclusions or you'll pull in machine-generated
# build output and mislabel it as human.
EXCLUDE_PATH_SUBSTRINGS = [
    "/test/", "/tests/", "/vendor/", "/vendored/", "/third_party/",
    "/node_modules/", "/generated/", "/build/", "/dist/", "/.git/",
    "_pb2.py", ".min.js", ".bundle.js", "-lock.json", "webpack.config",
]

GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# ---------------------------------------------------------------------------
# AI-code generation
# ---------------------------------------------------------------------------
# Multiple models so the cross-LLM generalization test (train on one
# model's outputs, test on another's) is possible. Add/remove as budget
# allows; each entry just needs a matching `call_<provider>` function in
# generate_ai_code.py.
AI_MODELS = [
    {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    # {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    # Add other providers once their API keys + network egress are
    # available in your own environment, e.g.:
    # {"provider": "openai", "model": "gpt-4o"},
]

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_VERSION = "2023-06-01"

GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 1024

PROBLEMS_PATH = ROOT_DIR / "prompts" / "problems.jsonl"

# For JS specifically, problems are tagged so generation can ask for either
# a frontend (React component) or backend (Node/Express) style solution,
# giving the JS class some internal stylistic diversity like the other
# three languages already have via varied problem types.
JS_VARIANTS = ["vanilla", "frontend_react", "backend_node"]
