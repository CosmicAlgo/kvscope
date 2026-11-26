"""Every repository path named in README.md must exist.

Checks two places:
  * inline `code` spans outside fenced code blocks that look like paths
    starting with a top-level folder of the repo; globs must match something;
  * the "Project structure" tree, rebuilt into full paths from its indentation.
"""

import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")

FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
INLINE = re.compile(r"`([^`\s]+)`")
TREE_ITEM = re.compile(r"^(?P<indent>(?:│   |    )*)[├└]── (?P<name>[^#]+?)\s*(?:#.*)?$")


def _read():
    with open(README, encoding="utf-8") as f:
        return f.read()


def _inline_paths(text):
    """Code spans that name a repo path: a "/"-path whose first part is a
    top-level entry of the repo.

    Skipped on purpose: bare filenames (``config.json`` usually names a file
    elsewhere) and HuggingFace ids such as ``EleutherAI/pythia-1.4b-deduped``."""
    top_level = set(os.listdir(ROOT))
    prose = FENCE.sub("", text)
    for token in INLINE.findall(prose):
        if token.startswith(("http://", "https://")):
            continue
        token = token.rstrip("/")
        if "/" in token and token.split("/")[0] in top_level:
            yield token


def _tree_paths(text):
    blocks = [b for b in FENCE.findall(text) if "\nkvscope/\n" in b]
    assert len(blocks) == 1, "expected exactly one project-structure tree"
    stack = []
    for line in blocks[0].splitlines():
        m = TREE_ITEM.match(line)
        if not m:
            continue
        depth = len(m.group("indent")) // 4
        stack = stack[:depth] + [m.group("name").rstrip("/")]
        yield "/".join(stack)


def _exists(rel):
    full = os.path.join(ROOT, *rel.split("/"))
    if any(ch in rel for ch in "*?["):
        return bool(glob.glob(full))
    return os.path.exists(full)


def test_tree_parser_sees_entries():
    paths = list(_tree_paths(_read()))
    assert "src/profiler/kv_tracer.py" in paths
    assert len(paths) > 20


def test_inline_paths_exist():
    missing = sorted({p for p in _inline_paths(_read()) if not _exists(p)})
    assert not missing, f"README.md names paths that do not exist: {missing}"


def test_tree_paths_exist():
    missing = [p for p in _tree_paths(_read()) if not _exists(p)]
    assert not missing, f"README.md project tree lists paths that do not exist: {missing}"
