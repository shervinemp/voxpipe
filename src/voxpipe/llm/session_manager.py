"""Centralized SessionManager for organizing, listing, saving, loading, and deleting sessions."""

import json
import os
import shutil
import time
import zipfile
from typing import Any, Dict, List, Optional

from voxpipe.core.utils import get_logger
from .model import LLM
from .session import Session


class SessionManager:
    """Centralized manager for isolated session directories, manifests, and session lifecycles."""

    def __init__(self, root_dir: str = "voxpipe_data"):
        self.logger = get_logger(__name__)
        self.root_dir = root_dir
        self.sessions_dir = os.path.join(root_dir, "sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)

    def save_session(
        self,
        session: Session,
        session_id: str,
        *,
        save_kv_cache: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save session into an isolated directory with manifest provenance."""
        session_path = os.path.join(self.sessions_dir, session_id)
        os.makedirs(session_path, exist_ok=True)

        # 1. Delegate core session save
        session.save(session_path, save_kv_cache=save_kv_cache)

        # 2. Write manifest.json
        manifest = {
            "session_id": session_id,
            "created_at": getattr(session, "_created_at", time.time()),
            "saved_at": time.time(),
            "llm_provider": type(session.llm).__name__,
            "max_turns": getattr(session.context_handler, "max_turns", 20),
            "custom": meta or {},
        }
        manifest_path = os.path.join(session_path, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info("Saved session '%s' to %s", session_id, session_path)
        return session_path

    def load_session(self, session_id: str, llm: LLM) -> Session:
        """Load a session by its unique ID."""
        session_path = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_path):
            raise FileNotFoundError(f"Session '{session_id}' not found in {self.sessions_dir}")

        session = Session.load(session_path, llm=llm)
        session_id_attr = session_id
        setattr(session, "session_id", session_id_attr)
        self.logger.info("Loaded session '%s' from %s", session_id, session_path)
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions and their manifest metadata."""
        sessions = []
        if not os.path.exists(self.sessions_dir):
            return sessions

        for s_id in os.listdir(self.sessions_dir):
            session_path = os.path.join(self.sessions_dir, s_id)
            if os.path.isdir(session_path):
                manifest_path = os.path.join(session_path, "manifest.json")
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path) as f:
                            sessions.append(json.load(f))
                    except Exception:
                        sessions.append({"session_id": s_id, "corrupted": True})
                else:
                    sessions.append({"session_id": s_id, "manifest": False})
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Safely delete an isolated session directory."""
        session_path = os.path.join(self.sessions_dir, session_id)
        if os.path.exists(session_path):
            shutil.rmtree(session_path)
            self.logger.info("Deleted session '%s'", session_id)
            return True
        return False

    def export_zip(self, session_id: str, zip_path: str) -> str:
        """Export an isolated session directory to a clean zip package."""
        session_path = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_path):
            raise FileNotFoundError(f"Session '{session_id}' not found")

        os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(session_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, session_path)
                    zf.write(full_path, arcname)

        self.logger.info("Exported session '%s' to zip %s", session_id, zip_path)
        return zip_path
