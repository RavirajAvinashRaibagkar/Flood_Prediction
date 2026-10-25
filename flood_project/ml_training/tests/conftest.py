# conftest.py — pytest configuration for FloodSentinel AI tests
import os
import sys
from pathlib import Path

# Add ml_training, backend, and project root to path so imports work
ML_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ML_DIR.parent / "backend"
ROOT_DIR = ML_DIR.parent.parent
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR))

# Suppress TF noise during tests
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
