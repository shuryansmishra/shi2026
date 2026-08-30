import os
import sys
from pathlib import Path

# Add project root and backend directories to Python sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"

for path in [str(backend_dir), str(root_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Default to serverless mock mode if not explicitly overridden
os.environ.setdefault("VQA_MOCK_MODE", "True")

try:
    from main import app
except ImportError:
    from backend.main import app

__all__ = ["app"]
