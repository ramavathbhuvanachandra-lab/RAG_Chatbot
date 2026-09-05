"""Phase 8B launcher.

Run from project root:

    python scripts/run_phase8b.py
"""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET = (
    ROOT
    / "tests"
    / "phase8"
    / "phase8b_real_world_eval.py"
)

if not TARGET.exists():
    raise SystemExit(
        f"Missing evaluator: {TARGET}"
    )

runpy.run_path(
    str(TARGET),
    run_name="__main__",
)
