"""
collect_human_code.py

Collects candidate *human-written* source files from GitHub for the
CodeTrust dataset, across Python, Java, C++, and JavaScript.

Why this is harder than "just download some code":
  - We need files that predate widespread AI coding assistants, so the
    "human" label is actually trustworthy (see config.HUMAN_CUTOFF_DATE).
  - We need permissively licensed repos so the dataset is redistributable.
  - We need to filter out vendored/generated/minified/test/lockfile files,
    which are extremely common in JS repos in particular and would poison
    the "human style" signal the model is supposed to learn.

Usage:
    export GITHUB_TOKEN=ghp_xxx        # strongly recommended (rate limits)
    python collect_human_code.py --language python --target 200
    python collect_human_code.py --language all --target 200

Output:
    data/human_code/<language>/<repo>__<path-slug>.<ext>
    data/human_code_metadata.jsonl   (one JSON record per saved file)
"""

import argparse
import base64
import json
import logging
import time
from pathlib import Path

import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("collect_human_code")

HEADERS = {"Accept": "application/vnd.github+json"}
if config.GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
else:
    log.warning(
        "No GITHUB_TOKEN set. GitHub's search API is heavily rate-limited "
        "(10 req/min) without auth -- collection will be slow. Set "
        "GITHUB_TOKEN in your environment for a much higher limit."
    )


def _request_with_backoff(url, params=None, max_retries=5):
    """GET with handling for GitHub's rate-limit and secondary-rate-limit responses."""
    for attempt in range(max_retries):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            return resp
        if resp.status_code in (403, 429):
            reset = resp.headers.get("X-RateLimit-Reset")
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                wait = int(retry_after) + 1
            elif reset:
                wait = max(int(reset) - int(time.time()), 1) + 1
            else:
                wait = 30 * (attempt + 1)
            log.warning("Rate limited (%s). Waiting %ss...", resp.status_code, wait)
            time.sleep(wait)
            continue
        log.warning("Request failed (%s) for %s: %s", resp.status_code, url, resp.text[:200])
        return resp
    return resp


def search_repos(language, cutoff_date, min_stars, page=1, per_page=30):
    """Find permissively-licensed repos created before cutoff_date with min_stars."""
    query = (
        f"language:{config.GITHUB_LANGUAGE_TAG[language]} "
        f"created:<{cutoff_date} "
        f"stars:>={min_stars} "
        f"archived:false"
    )
    resp = _request_with_backoff(
        f"{config.GITHUB_API_URL}/search/repositories",
        params={"q": query, "sort": "stars", "order": "desc", "page": page, "per_page": per_page},
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("items", [])


def get_repo_license(repo_full_name):
    resp = _request_with_backoff(f"{config.GITHUB_API_URL}/repos/{repo_full_name}/license")
    if resp.status_code != 200:
        return None
    return (resp.json().get("license") or {}).get("spdx_id", "").lower()


def get_default_branch_tree(repo_full_name, branch):
    resp = _request_with_backoff(
        f"{config.GITHUB_API_URL}/repos/{repo_full_name}/git/trees/{branch}",
        params={"recursive": "1"},
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("tree", [])


def is_candidate_path(path, extension):
    if not path.endswith(f".{extension}"):
        return False
    lower = path.lower()
    return not any(bad in lower for bad in config.EXCLUDE_PATH_SUBSTRINGS)


def get_last_commit_date(repo_full_name, path):
    resp = _request_with_backoff(
        f"{config.GITHUB_API_URL}/repos/{repo_full_name}/commits",
        params={"path": path, "per_page": 1},
    )
    if resp.status_code != 200 or not resp.json():
        return None
    return resp.json()[0]["commit"]["committer"]["date"]  # ISO 8601


def get_file_content(repo_full_name, path):
    resp = _request_with_backoff(f"{config.GITHUB_API_URL}/repos/{repo_full_name}/contents/{path}")
    if resp.status_code != 200:
        return None
    payload = resp.json()
    if payload.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(payload["content"]).decode("utf-8", errors="strict")
    except (ValueError, UnicodeDecodeError):
        return None


def slugify(path):
    return path.replace("/", "__")


def collect_for_language(language, target_count, cutoff_date, min_stars):
    log.info("Collecting human-written %s samples (target=%d)...", language, target_count)
    extension = config.EXTENSIONS[language]
    out_dir = config.HUMAN_DIR / language
    saved = 0
    page = 1
    metadata_records = []

    while saved < target_count and page <= 10:  # 10 pages ~= 300 repos, a sane ceiling
        repos = search_repos(language, cutoff_date, min_stars, page=page)
        if not repos:
            log.info("No more repos returned by search at page %d.", page)
            break

        for repo in repos:
            if saved >= target_count:
                break
            repo_full_name = repo["full_name"]

            license_id = get_repo_license(repo_full_name)
            if license_id not in config.ALLOWED_LICENSES:
                continue

            branch = repo.get("default_branch", "main")
            tree = get_default_branch_tree(repo_full_name, branch)
            candidates = [
                item["path"] for item in tree
                if item.get("type") == "blob" and is_candidate_path(item["path"], extension)
            ]
            if not candidates:
                continue

            # A couple of files per repo keeps author-diversity high in the dataset.
            for path in candidates[:3]:
                if saved >= target_count:
                    break

                last_commit = get_last_commit_date(repo_full_name, path)
                if not last_commit or last_commit[:10] >= cutoff_date:
                    continue  # file was touched too recently -- skip

                content = get_file_content(repo_full_name, path)
                if not content:
                    continue
                size = len(content.encode("utf-8"))
                if not (config.MIN_FILE_SIZE_BYTES <= size <= config.MAX_FILE_SIZE_BYTES):
                    continue

                out_path = out_dir / f"{slugify(repo_full_name)}__{slugify(path)}"
                out_path.write_text(content, encoding="utf-8")

                record = {
                    "id": f"human_{language}_{saved:05d}",
                    "language": language,
                    "label": "human_written",
                    "source_repo": repo_full_name,
                    "source_path": path,
                    "license": license_id,
                    "last_commit_date": last_commit,
                    "stars": repo.get("stargazers_count"),
                    "file_size_bytes": size,
                    "local_path": str(out_path),
                }
                metadata_records.append(record)
                saved += 1
                log.info("[%s] saved %d/%d: %s/%s", language, saved, target_count, repo_full_name, path)

        page += 1

    with open(config.HUMAN_METADATA_PATH, "a", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record) + "\n")

    log.info("Done with %s: saved %d samples.", language, saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Collect human-written code samples from GitHub.")
    parser.add_argument(
        "--language", choices=config.LANGUAGES + ["all"], default="all",
        help="Which language to collect. Default: all four.",
    )
    parser.add_argument(
        "--target", type=int, default=config.SAMPLES_PER_LANGUAGE_TARGET,
        help="Target number of samples per language.",
    )
    parser.add_argument("--cutoff-date", default=config.HUMAN_CUTOFF_DATE)
    parser.add_argument("--min-stars", type=int, default=config.MIN_REPO_STARS)
    args = parser.parse_args()

    languages = config.LANGUAGES if args.language == "all" else [args.language]
    totals = {}
    for lang in languages:
        totals[lang] = collect_for_language(lang, args.target, args.cutoff_date, args.min_stars)

    log.info("Summary: %s", totals)


if __name__ == "__main__":
    main()
