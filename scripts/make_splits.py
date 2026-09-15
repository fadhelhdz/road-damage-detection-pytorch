#!/usr/bin/env python3
"""Generate data/splits/czech_splits.json — the frozen, committed split.

Thin runner that puts src/ on the path (mirroring scripts/analyze_dataset.py)
and delegates to data.split. Run from the project root:

    python scripts/make_splits.py            # real Czech paths by default
    python scripts/make_splits.py --seed 42 --threshold 6
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from data.split import main  # noqa: E402

if __name__ == "__main__":
    main()
