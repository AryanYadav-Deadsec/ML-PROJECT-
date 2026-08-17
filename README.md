# CodeTrust Data Collection Pipeline

Collects a labeled human-vs-AI source code dataset across **Python, Java,
C++, and JavaScript** (frontend and backend JS both included under the
`javascript` label).

## Files

| File | Purpose |
|---|---|
| `config.py` | Languages, paths, cutoff dates, license allowlist, model list — edit this first |
| `collect_human_code.py` | Pulls verified pre-LLM human-written files from GitHub |
| `generate_ai_code.py` | Generates AI-written solutions via the Anthropic API |
| `prompts/problems.jsonl` | The problem set used to prompt AI generation (25 seed problems, 5 JS-specific) |
| `dataset_builder.py` | Merges both sources, de-dupes, writes train/val/test splits |

## Step-by-step

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set credentials
```bash
export GITHUB_TOKEN=ghp_xxxxxxxx        # github.com -> Settings -> Developer settings -> PAT
export ANTHROPIC_API_KEY=sk-ant-xxxxxxx
```
`GITHUB_TOKEN` isn't strictly required but without it the GitHub search API
is capped at 10 requests/minute, which makes collection painfully slow.

### 3. Collect human-written code
```bash
python collect_human_code.py --language all --target 500
```
This searches GitHub for permissively-licensed repos **created before
2021-01-01** (see `config.HUMAN_CUTOFF_DATE`), then additionally checks
that each individual file's **last commit** also predates that date. Both
checks matter — a repo can be old while still having recent, possibly
AI-assisted commits. Files under `test/`, `vendor/`, `node_modules/`,
`dist/`, minified/bundled JS, and lockfiles are skipped automatically —
this filter matters most for JavaScript, where build output otherwise
swamps real hand-written code.

Run one language at a time if you want to checkpoint progress:
```bash
python collect_human_code.py --language javascript --target 500
```

### 4. Generate AI-written code
```bash
python generate_ai_code.py --language all
```
For each of the 25 problems in `prompts/problems.jsonl`, this asks the
model(s) listed in `config.AI_MODELS` to solve it in the target language.
JavaScript problems carry a `js_variant` tag (`vanilla` /
`frontend_react` / `backend_node`) so the JS class contains a realistic
mix of plain scripts, React components, and Node backend code rather than
one narrow style.

To test cross-LLM generalization (PRD goal), add more models to
`config.AI_MODELS` and re-run — each model's output is tagged with its
name in the metadata, so you can later train on one model's outputs and
test on another's.

### 5. Build the final dataset
```bash
python dataset_builder.py
```
This merges both metadata files, drops exact-duplicate files (by content
hash — protects against, e.g., the same GitHub file showing up twice),
and writes stratified 70/15/15 train/val/test splits balanced across all
8 (language × label) groups. Outputs:
- `data/manifest.csv` — full labeled dataset, one row per file
- `data/splits/{train,val,test}.jsonl`

### 6. Sanity-check before training
Open `data/manifest.csv` and check the per-language, per-label counts
logged by `dataset_builder.py`. If any (language, label) group is small,
either lower `--target`'s cousin constraints (e.g. `--min-stars`) to
widen the GitHub search, or add more problems to `problems.jsonl` to
generate more AI samples for that language.

## Notes on scope decisions

- **Why is JS one label instead of "frontend" and "backend"?** The
  detector's job is human-vs-AI, not frontend-vs-backend — mixing styles
  within the JS class makes the model robust to *both* rather than
  overfitting to one. If you later want frontend/backend as a separate
  analysis axis, it's already there for free: every AI record carries
  `js_variant` in the metadata, and you can bucket human GitHub files by
  path (`/src/components/` etc. vs `/server/`, `/routes/`) with a small
  follow-up script.
- **Cutoff date honesty check**: `2021-01-01` is a heuristic, not a
  guarantee — some "human" files could still be AI-tab-completed via
  earlier tools (e.g. TabNine). Mention this as a known limitation in
  your report per the PRD's data-availability constraint.
- **Rate limits**: GitHub core API is 5000 req/hr authenticated; the
  *search* endpoint specifically is capped at 30 req/min authenticated.
  `collect_human_code.py` backs off automatically on 403/429, but for
  large targets (500+ per language) expect the human-code step to take
  a while — run it in the background or overnight.
