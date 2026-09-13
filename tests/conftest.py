import sys

from pathlib import Path

# src/ is the import root, so `import utils...` / `import evaluation...` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))