"""SessionManager: saves, lists, loads, deletes, exports and imports sessions on disk."""

import json
import os
import re
import shutil
import time
import zipfile
from typing import Any, Dict, List, Optional

from ..core.exceptions import VoxpipeError
from ..core.utils import get_logger
from .model import LLM
from .session import Session

_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MANIFEST_FILE = "manifest.json"
_STATE_FILE = "state.json"
_ARCHIVE_FILES = frozenset({_MANIFEST_FILE, _STATE_FILE, "conversation.json"})
_MAX_IMPORT_BYTES = 256 * 1024**2


class SessionManager:
    """Keeps each session in its own directory under ``<root_dir>/sessions``."""

    def __init__(self, root_dir: str = "voxpipe_data"):
        self.logger = get_logger(__name__)
        self.root_dir = root_dir
        self.sessions_dir = os.path.join(root_dir, "sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)

    def _session_path(self, session_id: str) -> str:
        """Directory for ``session_id``, refusing ids that could name anything outside the sessions folder."""
        if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id) or ".." in session_id:
            raise VoxpipeError(
                f"Invalid session id {session_id!r}: use 1-128 letters, digits, '.', '_' or '-', "
                "starting with a letter or digit."
            )
        return os.path.join(self.sessions_dir, session_id)

    def save_session(self, session: Session, session_id: str, *, meta: Optional[Dict[str, Any]] = None) -> str:
        """Save a session into its directory and write a manifest describing it."""
        session_path = self._session_path(session_id)
        os.makedirs(session_path, exist_ok=True)
        session.save(session_path)

        now = time.time()
        manifest_path = os.path.join(session_path, _MANIFEST_FILE)
        manifest = {
            "session_id": session_id,
            "created_at": self._read_manifest(manifest_path).get("created_at", now),
            "saved_at": now,
            "llm_provider": type(session.llm).__name__,
            "custom": meta or {},
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info("Saved session '%s' to %s", session_id, session_path)
        return session_path

    def load_session(self, session_id: str, llm: LLM, **session_kwargs) -> Session:
        """Load a saved session; keyword arguments are passed to the Session constructor."""
        session_path = self._session_path(session_id)
        if not os.path.isdir(session_path):
            raise FileNotFoundError(f"Session '{session_id}' not found in {self.sessions_dir}")

        session = Session.load(session_path, llm=llm, **session_kwargs)
        self.logger.info("Loaded session '%s' from %s", session_id, session_path)
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Manifest of every saved session; unreadable ones are flagged rather than skipped."""
        sessions = []
        for entry in sorted(os.listdir(self.sessions_dir)):
            session_path = os.path.join(self.sessions_dir, entry)
            if not os.path.isdir(session_path):
                continue
            manifest_path = os.path.join(session_path, _MANIFEST_FILE)
            if not os.path.exists(manifest_path):
                sessions.append({"session_id": entry, "manifest": False})
                continue
            manifest = self._read_manifest(manifest_path)
            sessions.append(manifest or {"session_id": entry, "corrupted": True})
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session's directory. Returns False if there was none."""
        session_path = self._session_path(session_id)
        if not os.path.isdir(session_path):
            return False
        shutil.rmtree(session_path)
        self.logger.info("Deleted session '%s'", session_id)
        return True

    def export_zip(self, session_id: str, zip_path: str) -> str:
        """Pack a session's files into a zip archive."""
        session_path = self._session_path(session_id)
        if not os.path.isdir(session_path):
            raise FileNotFoundError(f"Session '{session_id}' not found")

        os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in sorted(_ARCHIVE_FILES):
                file_path = os.path.join(session_path, name)
                if os.path.isfile(file_path):
                    zf.write(file_path, name)

        self.logger.info("Exported session '%s' to zip %s", session_id, zip_path)
        return zip_path

    def import_zip(self, zip_path: str, session_id: Optional[str] = None) -> str:
        """Import a session archive, replacing any session already stored under the id.

        The archive is treated as untrusted: only the known session files are
        extracted, its total size is capped, and any tool permissions or pending
        tool calls it carries are dropped, so importing a conversation can never
        pre-authorise an action.
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file '{zip_path}' not found")
        session_id = session_id or os.path.splitext(os.path.basename(zip_path))[0]
        session_path = self._session_path(session_id)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [info for info in zf.infolist() if not info.is_dir()]
            unexpected = sorted(info.filename for info in members if info.filename not in _ARCHIVE_FILES)
            if unexpected:
                raise VoxpipeError(f"Session archive contains unexpected files: {', '.join(unexpected)}")
            if sum(info.file_size for info in members) > _MAX_IMPORT_BYTES:
                raise VoxpipeError("Session archive is too large to import.")

            if os.path.isdir(session_path):
                shutil.rmtree(session_path)
            os.makedirs(session_path)
            zf.extractall(session_path, members=members)

        self._strip_tool_state(os.path.join(session_path, _STATE_FILE))
        self.logger.info("Imported zip '%s' into session '%s'", zip_path, session_id)
        return session_id

    @staticmethod
    def _strip_tool_state(state_path: str) -> None:
        if not os.path.exists(state_path):
            return
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.pop("tools", None) is not None:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

    @staticmethod
    def _read_manifest(manifest_path: str) -> Dict[str, Any]:
        """The manifest's contents, or an empty dict if it is missing or unreadable."""
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return manifest if isinstance(manifest, dict) else {}
