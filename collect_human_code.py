"""
collect_human_code.py

Collects candidate *human-written* source files from GitHub for the
CodeTrust dataset, across Python, Java, C++, and JavaScript.

Fixes vs your previous version (see chat for full diagnosis):
  - CRASH BUG: _request_with_backoff referenced `resp` after the retry loop
    even when every single attempt had raised a network exception (proxy
    hiccup, DNS blip, timeout) -- `resp` was never assigned, so the whole
    script died with UnboundLocalError instead of skipping and continuing.
    It now returns None on total failure, and every caller handles that.
  - GitHub's *secondary* (abuse-detection) rate limit fires on request
    *bursts* even when you're under your hourly quota -- this is very
    likely why you were seeing intermittent failures despite having a
    token. Added a fixed pacing delay between requests and a longer,
    dedicated backoff when a secondary-rate-limit response is detected.
  - Wasteful API usage: the old code fetched full file content just to
    check its size. The git tree API already reports each blob's size, so
    we filter on that first and only download content for files that will
    actually pass -- this alone cuts a large fraction of API calls.
  - RESUMABLE: previously, all progress was written to disk only once, at
    the very end of each language's run -- a crash or Ctrl-C midway lost
    everything and the next run started from page 1, burning through rate
    limits for nothing. Now: (a) metadata is appended+flushed after every
    single saved file, (b) already-saved (repo, path) pairs are skipped on
    a rerun, and (c) fully-scanned repos are checkpointed so reruns don't
    re-request their license/tree/commit-date over and over.
  - AUTO-RELAX: GitHub search returns at most 1000 results total. After
    applying the license + cutoff-date + path + size filters, that can
    yield far fewer usable files than --target, especially for Java/C++.
    If a language's search space is exhausted before hitting target, stars
    threshold is automatically halved (down to a floor) and search resumes,
    instead of silently stopping short with no explanation.

Usage:
    export GITHUB_TOKEN=ghp_xxx        # strongly recommended (rate limits)
    python collect_human_code.py --language python --target 1000
    python collect_human_code.py --language all --target 1000

Output:
    data/human_code/<language>/<repo>__<path-slug>.<ext>
    data/human_code_metadata.jsonl   (one JSON record per saved file, appended live)
    data/checkpoints/human_<language>.json   (resume state)
"""

import argparse
import base64
import json
import logging
import time

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


def _request_with_backoff(url, params=None, max_retries=None, is_search=False):
    """GET with handling for GitHub rate-limits and network errors.

    Returns the Response on success, or None if every retry was exhausted
    (caller must handle None -- never assume a Response comes back).
    """
    max_retries = max_retries or config.GITHUB_MAX_RETRIES
    resp = None
    for attempt in range(max_retries):
        # Pace every request so we don't trip GitHub's secondary/abuse
        # rate limit, which is based on burst rate, not just hourly quota.
        time.sleep(
            config.GITHUB_SEARCH_REQUEST_DELAY_SECONDS if is_search
            else config.GITHUB_REQUEST_DELAY_SECONDS
        )
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.exceptions.RequestException as err:
            wait = 10 * (attempt + 1)
            log.warning(
                "Network error (%s). Retrying in %ss... (attempt %d/%d)",
                err, wait, attempt + 1, max_retries,
            )
            time.sleep(wait)
            resp = None
            continue

        if resp.status_code == 200:
            return resp

        if resp.status_code in (403, 429):
            body_text = (resp.text or "").lower()
            is_secondary = "secondary rate limit" in body_text or "abuse" in body_text
            retry_after = resp.headers.get("Retry-After")
            reset = resp.headers.get("X-RateLimit-Reset")
            remaining = resp.headers.get("X-RateLimit-Remaining")

            if is_secondary:
                wait = config.GITHUB_SECONDARY_RATE_LIMIT_WAIT_SECONDS * (attempt + 1)
                log.warning("Hit GitHub's secondary rate limit. Waiting %ss...", wait)
            elif retry_after:
                wait = int(retry_after) + 1
            elif remaining == "0" and reset:
                wait = max(int(reset) - int(time.time()), 1) + 1
                log.warning("Primary rate limit exhausted. Waiting %ss until reset...", wait)
            else:
                wait = 30 * (attempt + 1)
                log.warning("Rate limited (%s). Waiting %ss...", resp.status_code, wait)
            time.sleep(wait)
            continue

        # Other errors (404, 422, 5xx, etc.) -- not worth retrying identically.
        log.warning("Request failed (%s) for %s: %s", resp.status_code, url, resp.text[:200])
        return resp

    if resp is None:
        log.error("Giving up on %s after %d attempts -- all were network errors.", url, max_retries)
    return resp


def search_repos(language, cutoff_date, min_stars, page=1, per_page=30):
    query = (
        f"language:{config.GITHUB_LANGUAGE_TAG[language]} "
        f"created:<{cutoff_date} "
        f"stars:>={min_stars} "
        f"archived:false"
    )
    resp = _request_with_backoff(
        f"{config.GITHUB_API_URL}/search/repositories",
        params={"q": query, "sort": "stars", "order": "desc", "page": page, "per_page": per_page},
        is_search=True,
    )
    if resp is None or resp.status_code != 200:
        return []
    return resp.json().get("items", [])


def get_repo_license(repo_full_name):
    resp = _request_with_backoff(f"{config.GITHUB_API_URL}/repos/{repo_full_name}/license")
    if resp is None or resp.status_code != 200:
        return None
    return (resp.json().get("license") or {}).get("spdx_id", "").lower()


def get_default_branch_tree(repo_full_name, branch):
    resp = _request_with_backoff(
        f"{config.GITHUB_API_URL}/repos/{repo_full_name}/git/trees/{branch}",
        params={"recursive": "1"},
    )
    if resp is None or resp.status_code != 200:
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
    if resp is None or resp.status_code != 200 or not resp.json():
        return None
    return resp.json()[0]["commit"]["committer"]["date"]  # ISO 8601


def get_file_content(repo_full_name, path):
    resp = _request_with_backoff(f"{config.GITHUB_API_URL}/repos/{repo_full_name}/contents/{path}")
    if resp is None or resp.status_code != 200:
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


# ---------------------------------------------------------------------------
# Resume / checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_path(language):
    return config.CHECKPOINT_DIR / f"human_{language}.json"


def load_checkpoint(language):
    path = checkpoint_path(language)
    if not path.exists():
        return {"visited_repos": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"visited_repos": []}


def save_checkpoint(language, state):
    checkpoint_path(language).write_text(json.dumps(state), encoding="utf-8")


def load_existing_human_records(language):
    """For resume: (repo, path) pairs already saved, and how many."""
    seen = set()
    count = 0
    if config.HUMAN_METADATA_PATH.exists():
        with open(config.HUMAN_METADATA_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("language") != language:
                    continue
                seen.add((rec["source_repo"], rec["source_path"]))
                count += 1
    return seen, count


def append_metadata(record):
    with open(config.HUMAN_METADATA_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def collect_for_language(language, target_count, cutoff_date, min_stars):
    log.info("Collecting human-written %s samples (target=%d)...", language, target_count)
    extension = config.EXTENSIONS[language]
    out_dir = config.HUMAN_DIR / language

    already_saved_pairs, saved = load_existing_human_records(language)
    if saved:
        log.info("Resuming: %d %s samples already saved from a previous run.", saved, language)

    checkpoint = load_checkpoint(language)
    visited_repos = set(checkpoint.get("visited_repos", []))

    current_min_stars = min_stars
    relax_round = 0

    while saved < target_count:
        page = 1
        exhausted_this_pass = False

        while saved < target_count and page <= 34:  # GitHub search caps at ~1000 results
            repos = search_repos(language, cutoff_date, current_min_stars, page=page)
            if not repos:
                exhausted_this_pass = True
                break

            for repo in repos:
                if saved >= target_count:
                    break
                repo_full_name = repo["full_name"]
                if repo_full_name in visited_repos:
                    continue  # already fully processed in a prior run

                license_id = get_repo_license(repo_full_name)
                if license_id not in config.ALLOWED_LICENSES:
                    visited_repos.add(repo_full_name)
                    continue

                branch = repo.get("default_branch", "main")
                tree = get_default_branch_tree(repo_full_name, branch)
                candidates = [
                    item for item in tree
                    if item.get("type") == "blob" and is_candidate_path(item["path"], extension)
                ]
                # Filter by size using the tree's own metadata BEFORE
                # spending an API call to download content.
                candidates = [
                    item for item in candidates
                    if item.get("size") is not None
                    and config.MIN_FILE_SIZE_BYTES <= item["size"] <= config.MAX_FILE_SIZE_BYTES
                ]
                if not candidates:
                    visited_repos.add(repo_full_name)
                    continue

                for item in candidates[:10]:
                    if saved >= target_count:
                        break
                    path = item["path"]
                    if (repo_full_name, path) in already_saved_pairs:
                        continue

                    last_commit = get_last_commit_date(repo_full_name, path)
                    if not last_commit or last_commit[:10] >= cutoff_date:
                        continue

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
                    append_metadata(record)
                    already_saved_pairs.add((repo_full_name, path))
                    saved += 1
                    log.info("[%s] saved %d/%d: %s/%s", language, saved, target_count, repo_full_name, path)

                visited_repos.add(repo_full_name)
                # Persist checkpoint periodically so a crash doesn't force
                # re-scanning repos we already paid API calls to evaluate.
                save_checkpoint(language, {"visited_repos": sorted(visited_repos)})

            page += 1

        if saved >= target_count:
            break

        if exhausted_this_pass or page > 34:
            relax_round += 1
            if relax_round > config.MAX_RELAX_ROUNDS or current_min_stars <= config.MIN_STARS_FLOOR:
                log.warning(
                    "Exhausted GitHub's searchable pool for %s at min_stars=%d "
                    "with only %d/%d samples. Stopping short of target -- "
                    "consider adding more source repos manually, or accept a "
                    "smaller dataset for this language.",
                    language, current_min_stars, saved, target_count,
                )
                break
            new_min_stars = max(config.MIN_STARS_FLOOR, int(current_min_stars * config.MIN_STARS_RELAX_FACTOR))
            log.warning(
                "Search space exhausted for %s at min_stars=%d (saved %d/%d). "
                "Relaxing min_stars to %d and continuing...",
                language, current_min_stars, saved, target_count, new_min_stars,
            )
            current_min_stars = new_min_stars

    log.info("Done with %s: saved %d/%d samples.", language, saved, target_count)
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
