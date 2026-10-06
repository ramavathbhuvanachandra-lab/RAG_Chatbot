"""Remove legacy application dependencies from the ai_platform package.

Run this from the repository root after copying the standalone runtime/core
adapter files into place. The script edits only files below ``ai_platform/``.
The old ``backend/`` package is never modified.
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "ai_platform"

# Longest/specific mappings first so root-level legacy imports land at the
# intended new platform boundary instead of a guessed sibling module.
REPLACEMENTS = (
    ("backend.conversation_resolver", "ai_platform.core.conversation"),
    ("backend.multi_intent", "ai_platform.core.multi_intent"),
    ("backend.retriever", "ai_platform.runtime.retrieval"),
    ("backend.ingestion", "ai_platform.runtime.corpus"),
    ("backend.vectorstore", "ai_platform.runtime.vectorstore"),
    ("backend.embedding", "ai_platform.runtime.embedding"),
    ("backend.llm", "ai_platform.runtime.llm"),
    ("backend.config", "ai_platform.runtime.config"),
    ("backend.core", "ai_platform.core"),
    ("backend.institutions", "ai_platform.institutions"),
    ("backend.runtime", "ai_platform.runtime"),
)


def patch_text(text: str) -> str:
    updated = text
    for old, new in REPLACEMENTS:
        updated = updated.replace(old, new)

    # Current CoreNodes uses a legacy-provider fallback for answer_llm.
    # Replace the complete fallback with one platform-only provider.
    old_block = '''    def answer_model_provider(_state: State) -> Any:\n        try:\n            return getattr(\n                importlib.import_module("ai_platform.runtime.llm"),\n                "answer_llm",\n            )\n        except (ImportError, AttributeError):\n            return getattr(\n                importlib.import_module("ai_platform.runtime.llm"),\n                "answer_llm",\n            )\n'''
    new_block = '''    def answer_model_provider(_state: State) -> Any:\n        return getattr(\n            importlib.import_module("ai_platform.runtime.llm"),\n            "answer_llm",\n        )\n'''
    updated = updated.replace(old_block, new_block)

    # Older snapshots can still carry direct backend.core imports in strings.
    # The mapping above already handles the known paths; this catches any
    # remaining module-qualified references without touching the legacy tree.
    updated = re.sub(r"\\bbackend\\.core\\b", "ai_platform.core", updated)
    updated = re.sub(r"\\bbackend\\.institutions\\b", "ai_platform.institutions", updated)
    updated = re.sub(r"\\bbackend\\.runtime\\b", "ai_platform.runtime", updated)

    return updated


def main() -> None:
    if not PLATFORM.is_dir():
        raise SystemExit(f"Missing ai_platform directory: {PLATFORM}")

    changed: list[str] = []
    for path in sorted(PLATFORM.rglob("*.py")):
        original = path.read_text(encoding="utf-8")
        updated = patch_text(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(str(path.relative_to(ROOT)))

    print("Patched files:")
    for item in changed:
        print(f"  {item}")
    print(f"Total: {len(changed)}")


if __name__ == "__main__":
    main()
