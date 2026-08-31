"""
feature_extraction.py

Computes the handcrafted stylistic/structural feature vector used by the
baseline model (PRD section 5, requirement 3). One extractor, shared
across all four languages, so the feature schema is identical regardless
of which language a sample is -- this is what lets a single combined
model (see baseline_model.py --per-language flag) work at all.

Design note: this uses regex/heuristic parsing for all languages, with
extra precision layered on for Python via the stdlib `ast` module where it
parses cleanly. Full AST parsers per language (javalang, pycparser/clang,
tree-sitter) are the more "correct" long-term approach flagged in
ARCHITECTURE.md section 2.2 -- this regex-based version is the pragmatic
starting point that needs zero extra native dependencies. Swap in real
per-language parsers later without changing anything downstream, since
baseline_model.py only cares about the returned dict's keys/values.
"""

import ast
import math
import re
from collections import Counter

LINE_COMMENT = {
    "python": "#",
    "java": "//",
    "cpp": "//",
    "javascript": "//",
}

BLOCK_COMMENT = {
    "python": [('"""', '"""'), ("'''", "'''")],
    "java": [("/*", "*/")],
    "cpp": [("/*", "*/")],
    "javascript": [("/*", "*/")],
}

FUNC_PATTERNS = {
    "python": re.compile(r"^\s*(?:async\s+)?def\s+\w+\s*\(", re.MULTILINE),
    "java": re.compile(
        r"\b(?:public|private|protected|static|final|synchronized|\s)+"
        r"[\w<>\[\],\s]+\s+\w+\s*\([^;{]*\)\s*\{"
    ),
    "cpp": re.compile(r"\b[\w:<>~,\s\*&]+\s+\w+\s*\([^;{]*\)\s*\{"),
    "javascript": re.compile(
        r"\bfunction\s*\w*\s*\(|=>\s*\{|=>\s*[^\{]|"
        r"\b\w+\s*\([^)]*\)\s*\{(?=[^}]*return)"
    ),
}

CONTROL_FLOW_KEYWORDS = {
    "python": ["if", "elif", "for", "while", "except", "and", "or"],
    "java": ["if", "for", "while", "case", "catch", "&&", "\\|\\|", "\\?"],
    "cpp": ["if", "for", "while", "case", "catch", "&&", "\\|\\|", "\\?"],
    "javascript": ["if", "for", "while", "case", "catch", "&&", "\\|\\|", "\\?"],
}

GENERIC_NAMES = {
    "i", "j", "k", "x", "y", "z", "tmp", "temp", "data", "result", "res",
    "val", "value", "item", "obj", "arr", "ret", "foo", "bar", "baz",
    "n", "m", "a", "b", "idx", "index", "list", "str", "num", "output",
}

BOILERPLATE_COMMENT_PHRASES = [
    "this function", "helper function", "returns the", "example usage",
    "initialize", "todo", "fixme", "note:", "this method", "handles the",
]

IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")
CAMEL_CASE_RE = re.compile(r"^[a-z][a-z0-9]*([A-Z][a-z0-9]*)+$")


def _shannon_entropy(items):
    if not items:
        return 0.0
    counts = Counter(items)
    total = len(items)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _strip_string_and_comment_noise(code):
    """Rough string-literal stripper so keyword/identifier counts aren't
    thrown off by control-flow words appearing inside string contents."""
    return re.sub(r"(\".*?\"|'.*?')", '""', code)


def _regex_features(code, language):
    lines = code.splitlines()
    num_lines = max(len(lines), 1)
    blank_lines = sum(1 for l in lines if not l.strip())
    line_lengths = [len(l) for l in lines] or [0]

    line_marker = LINE_COMMENT.get(language, "//")
    comment_lines = sum(1 for l in lines if l.strip().startswith(line_marker))
    for start, end in BLOCK_COMMENT.get(language, []):
        comment_lines += code.count(start)  # rough: count block-comment openings as extra "comment lines"

    trailing_ws_lines = sum(1 for l in lines if l != l.rstrip() and l.strip())

    code_no_strings = _strip_string_and_comment_noise(code)

    func_pattern = FUNC_PATTERNS.get(language)
    num_functions = len(func_pattern.findall(code_no_strings)) if func_pattern else 0

    keywords = CONTROL_FLOW_KEYWORDS.get(language, [])
    complexity = 1
    for kw in keywords:
        complexity += len(re.findall(rf"\b{kw}\b" if kw.isalpha() else kw, code_no_strings))

    # Brace/indent-based nesting depth approximation.
    if language == "python":
        depths = []
        for l in lines:
            stripped = l.lstrip(" ")
            if not stripped or stripped.startswith(line_marker):
                continue
            indent = len(l) - len(stripped)
            depths.append(indent // 4)
        max_nesting = max(depths) if depths else 0
    else:
        depth, max_depth = 0, 0
        for ch in code_no_strings:
            if ch == "{":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == "}":
                depth = max(depth - 1, 0)
        max_nesting = max_depth

    identifiers = [
        m for m in IDENTIFIER_RE.findall(code_no_strings)
        if not m.isupper() or len(m) > 1  # drop stray single uppercase letters (type params etc.)
    ]
    generic_count = sum(1 for ident in identifiers if ident.lower() in GENERIC_NAMES)
    snake_count = sum(1 for ident in identifiers if SNAKE_CASE_RE.match(ident))
    camel_count = sum(1 for ident in identifiers if CAMEL_CASE_RE.match(ident))

    comment_text_lower = "\n".join(
        l.strip() for l in lines if l.strip().startswith(line_marker)
    ).lower()
    boilerplate_hits = sum(comment_text_lower.count(p) for p in BOILERPLATE_COMMENT_PHRASES)

    semicolon_lines = sum(1 for l in lines if l.strip().endswith(";"))
    string_literal_count = len(re.findall(r"\".*?\"|'.*?'", code))

    return {
        "num_lines": num_lines,
        "avg_line_length": sum(line_lengths) / num_lines,
        "max_line_length": max(line_lengths),
        "blank_line_ratio": blank_lines / num_lines,
        "comment_density": comment_lines / num_lines,
        "num_functions": num_functions,
        "avg_function_length": num_lines / max(num_functions, 1),
        "max_nesting_depth": max_nesting,
        "cyclomatic_complexity_approx": complexity,
        "num_identifiers": len(identifiers),
        "avg_identifier_length": (sum(len(i) for i in identifiers) / len(identifiers)) if identifiers else 0.0,
        "identifier_naming_entropy": _shannon_entropy(identifiers),
        "generic_name_ratio": generic_count / len(identifiers) if identifiers else 0.0,
        "snake_case_ratio": snake_count / len(identifiers) if identifiers else 0.0,
        "camel_case_ratio": camel_count / len(identifiers) if identifiers else 0.0,
        "boilerplate_comment_hits": boilerplate_hits,
        "trailing_whitespace_ratio": trailing_ws_lines / num_lines,
        "semicolon_line_ratio": semicolon_lines / num_lines,
        "string_literal_count": string_literal_count,
    }


def _python_ast_overrides(code):
    """More precise versions of a few features, when the code parses as
    valid Python. Returns {} (no overrides) if it doesn't parse -- callers
    fall back to the regex-based estimates in that case."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return {}

    func_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def node_depth(node, depth=0):
        child_depths = [
            node_depth(child, depth + 1)
            for child in ast.iter_child_nodes(node)
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With))
        ]
        return max(child_depths, default=depth)

    max_nesting = max((node_depth(n) for n in ast.walk(tree)
                        if isinstance(n, (ast.If, ast.For, ast.While))), default=0)

    complexity = 1 + sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.ExceptHandler))
    )

    docstring_count = sum(
        1 for n in ([tree] + func_nodes) if ast.get_docstring(n)
    )
    docstring_ratio = docstring_count / max(len(func_nodes), 1)

    return {
        "num_functions": len(func_nodes),
        "max_nesting_depth": max_nesting,
        "cyclomatic_complexity_approx": complexity,
        "docstring_ratio": docstring_ratio,
    }


def extract_features(code, language):
    """Returns a flat dict[str, float] of stylistic features for `code`.
    Same schema regardless of `language`, so a single combined model can
    consume output from any of the four supported languages."""
    if not code or not code.strip():
        code = " "  # avoid div-by-zero paths on an empty file
    features = _regex_features(code, language)
    features.setdefault("docstring_ratio", 0.0)

    if language == "python":
        features.update(_python_ast_overrides(code))

    return {k: float(v) for k, v in features.items()}


FEATURE_NAMES = sorted(extract_features("def f():\n    pass\n", "python").keys())
