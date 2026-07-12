import sys
from pathlib import Path

# Add the project root to sys.path so tests can import from src/
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
