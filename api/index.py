import sys
from pathlib import Path

# Add backend directory to Python sys.path so all imports work seamlessly
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import app
