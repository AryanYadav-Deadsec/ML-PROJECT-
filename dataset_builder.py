"""
dataset_builder.py

Merges data/human_code_metadata.jsonl + data/ai_code_metadata.jsonl into
one manifest, de-duplicates by content hash, and writes stratified
train/val/test splits (stratified by language x label so every split has
a balanced mix across all four languages).

Usage:
    python dataset_builder.py
"""

import csv
import hashlib
import json
import logging
import random

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dataset_builder")

random.seed(42)

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.15, 0.15


def load_jsonl(path):
    records = []
    if not path.exists():
        log.warning("%s does not exist yet -- run the collection scripts first.", path)
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def content_hash(local_path):
    try:
        with open(local_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return None


def dedupe(records):
    seen = set()
    deduped = []
    for r in records:
        h = content_hash(r["local_path"])
        if h is None:
            continue
        if h in seen:
            continue
        seen.add(h)
        r["content_sha256"] = h
        deduped.append(r)
    return deduped


def stratified_split(records):
    """Split within each (language, label) group so every split is balanced."""
    groups = {}
    for r in records:
        key = (r["language"], r["label"])
        groups.setdefault(key, []).append(r)

    train, val, test = [], [], []
    for key, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
        log.info("%s: %d total -> train=%d val=%d test=%d", key, n, n_train, n_val, n - n_train - n_val)

    return train, val, test


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_manifest_csv(records, path):
    fieldnames = sorted({k for r in records for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def main():
    human = load_jsonl(config.HUMAN_METADATA_PATH)
    ai = load_jsonl(config.AI_METADATA_PATH)
    log.info("Loaded %d human records, %d AI records.", len(human), len(ai))

    all_records = dedupe(human + ai)
    log.info("After de-dup by content hash: %d records.", len(all_records))

    # Summary counts per language/label -- sanity check before training.
    summary = {}
    for r in all_records:
        key = (r["language"], r["label"])
        summary[key] = summary.get(key, 0) + 1
    for key, count in sorted(summary.items()):
        log.info("  %s: %d", key, count)

    write_manifest_csv(all_records, config.MANIFEST_PATH)

    train, val, test = stratified_split(all_records)
    write_jsonl(train, config.SPLITS_DIR / "train.jsonl")
    write_jsonl(val, config.SPLITS_DIR / "val.jsonl")
    write_jsonl(test, config.SPLITS_DIR / "test.jsonl")

    log.info(
        "Wrote manifest (%s) and splits: train=%d val=%d test=%d",
        config.MANIFEST_PATH, len(train), len(val), len(test),
    )


if __name__ == "__main__":
    main()
