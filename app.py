"""
Root-level server/app.py — required by openenv validate.
Re-exports the FastAPI app from the environment package.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from envs.traffic_signal_env.server.app import app  # noqa: F401

__all__ = ["app"]
