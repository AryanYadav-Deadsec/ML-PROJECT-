# CodeTrust — Architecture

Companion to `PRD_v2.md`. Describes how the system is actually built,
component by component, in the order they run.

## 1. System overview

```
                     ┌─────────────────────┐        ┌──────────────────────┐
                     │   GitHub REST API   │        │  Anthropic Messages  │
                     │  (human code source)│        │  API (AI code source)│
                     └──────────┬───────────┘        └──────────┬───────────┘
                                │                                │
                     ┌──────────▼───────────┐        ┌──────────▼───────────┐
                     │ collect_human_code.py│        │ generate_ai_code.py  │
                     └──────────┬───────────┘        └──────────┬───────────┘
                                └───────────────┬────────────────┘
                                                 ▼
                                     ┌───────────────────────┐
                                     │   dataset_builder.py   │
                                     │  dedupe + stratified   │
                                     │   train/val/test split │
                                     └───────────┬─────────────┘
                                                 ▼
                          ┌──────────────────────────────────────┐
                          │        data/splits/*.jsonl            │
                          └───────────────────┬────────────────────┘
                                               │
                 ┌─────────────────────────────┼─────────────────────────────┐
                 ▼                             ▼                             ▼
      ┌───────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
      │ feature_extraction │      │   codebert_train.py   │      │ graphcodebert_train.py│
      │  (AST + stylistic) │      │  (token-level model)  │      │ (+ data-flow graph)   │
      └─────────┬───────────┘      └──────────┬─────────────┘      └──────────┬─────────────┘
                 ▼                             ▼                             ▼
      ┌───────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
      │ baseline_model.py  │      │  codebert_finetuned/  │      │graphcodebert_finetuned│
      │ (LogReg / RF)      │      │      (saved model)    │      │     (saved model)     │
      └─────────┬───────────┘      └──────────┬─────────────┘      └──────────┬─────────────┘
                 └─────────────────────────────┼─────────────────────────────┘
                                                 ▼
                                     ┌───────────────────────┐
                                     │      eval_suite.py     │
                                     │ standard / cross-LLM /  │
                                     │  evasion robustness     │
                                     └───────────┬─────────────┘
                                                 ▼
                          ┌──────────────────────────────────────┐
                          │         FastAPI backend (api/)         │
                          │   loads best model(s) at startup;      │
                          │   POST /predict → label + explanation  │
                          └───────────────────┬────────────────────┘
                                               ▼
                          ┌──────────────────────────────────────┐
                          │         React frontend (ui/)           │
                          │  upload/paste → calls /predict →       │
                          │  renders label, confidence, highlights │
                          └────────────────────────────────────────┘
```

Everything left of `data/splits/*.jsonl` is the **data-collection stage**
(already built — see `data_collection/README.md`). Everything right of it
is the **modeling stage** (not yet built; this doc specifies it so it can
be implemented next).

## 2. Component breakdown

### 2.1 Data collection (built)

| Component | Responsibility |
|---|---|
| `config.py` | Single source of truth for languages, paths, cutoff date, license allowlist, model list |
| `collect_human_code.py` | GitHub search → license/date/path filtering → saves raw files + metadata |
| `generate_ai_code.py` | Problem set → LLM API calls → saves generated files + metadata (model, variant) |
| `dataset_builder.py` | Merge, dedupe (content hash), stratified split by (language × label) |

Output contract: every downstream stage consumes `data/splits/{train,val,test}.jsonl`,
where each record is `{id, local_path, language, label, js_variant?, source_model?}`.
Nothing downstream should need to know whether a sample came from GitHub or
an LLM API — that's exactly what this contract is for.

### 2.2 Feature extraction (`feature_extraction/`, to build)

Produces the handcrafted feature vector consumed by the baseline model and
by SHAP/LIME. One extractor module per language, all conforming to the
same output schema (`dict[str, float]`) so the baseline model doesn't care
which language produced it:

| Language | AST tool | Notes |
|---|---|---|
| Python | `ast` (stdlib) | Most direct — build first, use it to validate the schema |
| Java | `javalang` | Pure Python, adequate for a course-project timeline |
| C++ | `pycparser` or `clang` bindings | `pycparser` needs preprocessed input (no real includes); `clang` bindings are heavier to set up but handle real-world C++ headers. Pick `pycparser` first, note the limitation in the report, upgrade only if time allows. |
| JavaScript | `esprima` or `tree-sitter-javascript` | **Recommendation: `tree-sitter-javascript`.** `esprima` is simpler to install but stalls on JSX and newer ES syntax, which is a real risk given the `frontend_react` variant in the dataset. `tree-sitter` handles JSX/TS-adjacent syntax cleanly and is the more future-proof choice, at the cost of one extra native-binding install step. |

Feature set (language-agnostic where possible): nesting depth, cyclomatic
complexity, identifier-naming entropy, comment density, average
line/function length, ratio of boilerplate patterns (e.g. generic
`temp`/`data`/`result` variable names), token-type distribution. Add 2-3
JS-specific features once the extractor is working: use of `var` vs
`let`/`const` (AI-generated JS skews modern), arrow-function ratio, and
(for the frontend variant) JSX-prop-count distribution.

### 2.3 Baseline model (`baseline_model.py`, to build)

Logistic Regression and Random Forest over the feature vectors from §2.2,
one model trained per language (or one multi-language model with a
`language` one-hot feature — try both, keep whichever gets a better
held-out F1). SHAP (preferred) or LIME wraps the trained model directly;
scikit-learn's native feature importances are not sufficient per the PRD's
explainability requirement, since they don't give a per-prediction
explanation, only a global one.

### 2.4 Transformer models (`codebert_train.py`, `graphcodebert_train.py`, to build)

- Both fine-tune a Hugging Face `AutoModelForSequenceClassification` head
  on top of the pretrained CodeBERT / GraphCodeBERT checkpoints, binary
  classification.
- GraphCodeBERT additionally needs the data-flow graph as input — Microsoft's
  own preprocessing script (from the GraphCodeBERT repo) handles this for
  Python/Java/C/C++/JavaScript already, so no custom data-flow extraction
  is needed; reuse it rather than reimplementing.
- Train Python first end-to-end (tokenize → fine-tune → eval → explain)
  before extending to the other three languages — this validates the whole
  pipeline against the PRD's F1 > 0.90 target on the language with the most
  data, before spending compute budget elsewhere.
- Attention-weight extraction: use `output_attentions=True` at inference
  time, average over the last layer's heads, map attention weight back to
  the originating token span for the highlight explanation.

### 2.5 Evaluation suite (`eval_suite.py`, to build)

Three modes, all reading from `data/splits/test.jsonl` plus the metadata
files for model/source info:

1. **Standard**: accuracy/F1/precision/recall per language, per model.
2. **Cross-LLM generalization**: filter AI samples by `source_model`,
   train (or re-evaluate) on model A's outputs, test on model B's — this
   is why `generate_ai_code.py` tags every sample with its generating
   model.
3. **Evasion robustness**: apply variable renaming, reformatting, and
   comment stripping to test-set samples on the fly (simple AST-based
   transforms per language, reusing the extractors from §2.2), re-run
   inference, report the accuracy delta.

### 2.6 API (`api/`, to build)

FastAPI app, single `POST /predict` route matching the contract in
PRD §6.1. Loads the best-performing model per the PRD's `model_used` field
at startup (config-driven, not hardcoded) so swapping in a newer
checkpoint doesn't require a code change. No auth/rate-limiting per PRD
§4 non-goals.

### 2.7 Web UI (`ui/`, to build)

Single React page: language selector (or auto-detect from uploaded file
extension, including `.js`/`.jsx`), code textarea or file upload, submit
button, results panel showing label + confidence + explanation
(highlighted spans for transformer results, a bar chart of feature
attributions for baseline results). No accounts, no history, per PRD §4.

## 3. JavaScript-specific handling (why it needed its own thread here)

JS is architecturally the odd one out among the four languages, for two
reasons worth designing around explicitly rather than discovering mid-build:

1. **No stdlib AST module.** Python/Java/C++ each have an established
   single library choice; JS doesn't, hence the `esprima` vs `tree-sitter`
   decision in §2.2. Make this call early — switching AST libraries after
   the feature extractor is half-written means redoing the feature schema.
2. **Two structurally different sub-styles under one label.** A React
   component and an Express route handler look nothing alike
   syntactically, but both get the label `javascript`. The `js_variant`
   metadata tag (§2.1 of the PRD) exists so this can be measured
   (§7 of the PRD — F1 broken out by variant) without it becoming a second
   modeling target. If the variant-level F1 turns out to be lopsided
   (e.g. great on backend, weak on frontend), that becomes a named
   limitation in the report rather than a silent gap.

## 4. Build order (recommended)

1. Data collection — **done**.
2. Python feature extractor + baseline model, end-to-end through SHAP
   explanations. Smallest surface area, validates the whole
   extract → train → explain loop once.
3. Python CodeBERT fine-tune + attention highlighting. Validates the
   transformer loop once, same reasoning.
4. `eval_suite.py` against the Python models — confirms the F1 > 0.90
   target is reachable before sinking time into the other three languages.
5. Extend feature extraction + both model types to Java, C++, then
   JavaScript, in that order (JS last since its tooling decision in §2.2
   carries the most risk).
6. GraphCodeBERT for Python, then other languages if time allows (PRD §8).
7. Cross-LLM and evasion evaluation modes.
8. API, then UI (UI depends on the API contract being stable).
9. Final report, folding in the data-provenance caveats from PRD §3.1 and
   the JS-variant breakdown from PRD §7.

## 5. Open decisions to resolve before coding §2.2 onward

- `pycparser` vs `clang` bindings for C++ (affects how much real-world C++
  the extractor can actually parse).
- `esprima` vs `tree-sitter-javascript` for JS (affects JSX/modern-syntax
  coverage).
- One multi-language baseline model vs. one-per-language (affects §2.3;
  worth a quick experiment rather than a guess).
