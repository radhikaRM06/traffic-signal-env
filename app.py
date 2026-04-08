"""
HuggingFace Space entry point.
Imports and re-exports the FastAPI app for uvicorn.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Re-export app so uvicorn can find it at the top level:
#   uvicorn app:app --host 0.0.0.0 --port 7860
from envs.traffic_signal_env.server.app import app  # noqa: F401
