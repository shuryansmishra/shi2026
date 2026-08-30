import os
import sys
from pathlib import Path

# Add backend directory to sys.path so internal imports inside backend work at runtime
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"

for p in [str(backend_dir), str(root_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("VQA_MOCK_MODE", "True")

# Direct import from backend package (fixes IDE unresolved import warning)
from backend.main import app

__all__ = ["app"]
