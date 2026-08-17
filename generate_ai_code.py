"""
generate_ai_code.py

Generates the *AI-written* half of the CodeTrust dataset by prompting one
or more LLMs (config.AI_MODELS) to solve each problem in prompts/problems.jsonl,
in each of the four target languages.

JavaScript gets extra treatment: problems tagged with a "js_variant"
(vanilla / frontend_react / backend_node) are only generated in that
style, so the JS class mixes plain scripts, React components, and Node
backend code -- matching the real-world mix you're trying to detect,
instead of one narrow JS "flavor".

Usage:
    export ANTHROPIC_API_KEY=sk-ant-xxx
    python generate_ai_code.py --language python
    python generate_ai_code.py --language all

Output:
    data/ai_code/<language>/<model>__<problem_id>.<ext>
    data/ai_code_metadata.jsonl
"""

import argparse
import json
import logging
import re
import time

import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate_ai_code")

CODE_FENCE_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

LANGUAGE_PROMPT_HINT = {
    "python": "Write idiomatic Python 3.",
    "java": "Write idiomatic Java (a single public class is fine).",
    "cpp": "Write idiomatic modern C++ (C++17).",
}

JS_VARIANT_PROMPT_HINT = {
    "vanilla": "Write plain modern JavaScript (ES2020+), no framework.",
    "frontend_react": "Write a functional React component using hooks.",
    "backend_node": "Write Node.js backend code (CommonJS or ESM, your choice).",
}


def load_problems():
    problems = []
    with open(config.PROBLEMS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def build_prompt(problem, language):
    if language == "javascript":
        variant = problem.get("js_variant", "vanilla")
        style_hint = JS_VARIANT_PROMPT_HINT[variant]
    else:
        style_hint = LANGUAGE_PROMPT_HINT[language]

    return (
        f"{style_hint}\n\n"
        f"Task: {problem['description']}\n\n"
        "Return ONLY the code in a single fenced code block, with no "
        "explanation before or after it."
    )


def extract_code(response_text):
    match = CODE_FENCE_RE.search(response_text)
    return match.group(1).strip() if match else response_text.strip()


def call_anthropic(prompt, model):
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
    headers = {
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": config.ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": config.GENERATION_MAX_TOKENS,
        "temperature": config.GENERATION_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(config.ANTHROPIC_API_URL, headers=headers, json=body, timeout=60)
    if resp.status_code == 429:
        wait = int(resp.headers.get("retry-after", 20))
        log.warning("Rate limited by Anthropic API, waiting %ss...", wait)
        time.sleep(wait)
        return call_anthropic(prompt, model)
    resp.raise_for_status()
    data = resp.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return "\n".join(text_blocks)


PROVIDER_FUNCS = {
    "anthropic": call_anthropic,
    # "openai": call_openai,   # implement + add to config.AI_MODELS if you
    #                          # have OpenAI access and it's allowed on your
    #                          # network egress list.
}


def generate_for_language(language, problems, model_cfg):
    provider, model = model_cfg["provider"], model_cfg["model"]
    call_fn = PROVIDER_FUNCS.get(provider)
    if call_fn is None:
        log.warning("No call function implemented for provider '%s', skipping.", provider)
        return 0

    extension = config.EXTENSIONS[language]
    out_dir = config.AI_DIR / language
    saved = 0
    metadata_records = []

    applicable_problems = problems
    if language == "javascript":
        # every problem is usable for JS: generic ones default to "vanilla"
        applicable_problems = problems

    for problem in applicable_problems:
        prompt = build_prompt(problem, language)
        try:
            raw_response = call_fn(prompt, model)
        except Exception as exc:  # noqa: BLE001 - log and keep going
            log.error("Generation failed for %s/%s: %s", language, problem["problem_id"], exc)
            continue

        code = extract_code(raw_response)
        if not code:
            continue

        file_name = f"{model.replace('/', '_')}__{problem['problem_id']}.{extension}"
        out_path = out_dir / file_name
        out_path.write_text(code, encoding="utf-8")

        record = {
            "id": f"ai_{language}_{model}_{problem['problem_id']}",
            "language": language,
            "label": "ai_generated",
            "provider": provider,
            "model": model,
            "problem_id": problem["problem_id"],
            "js_variant": problem.get("js_variant") if language == "javascript" else None,
            "temperature": config.GENERATION_TEMPERATURE,
            "local_path": str(out_path),
        }
        metadata_records.append(record)
        saved += 1
        log.info("[%s/%s] generated %s", language, model, problem["problem_id"])

    with open(config.AI_METADATA_PATH, "a", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record) + "\n")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate AI-written code samples.")
    parser.add_argument("--language", choices=config.LANGUAGES + ["all"], default="all")
    args = parser.parse_args()

    languages = config.LANGUAGES if args.language == "all" else [args.language]
    problems = load_problems()
    log.info("Loaded %d problems.", len(problems))

    totals = {}
    for language in languages:
        total = 0
        for model_cfg in config.AI_MODELS:
            total += generate_for_language(language, problems, model_cfg)
        totals[language] = total

    log.info("Summary: %s", totals)


if __name__ == "__main__":
    main()
