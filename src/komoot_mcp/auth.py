"""Komoot authentication — Basic Auth (email/password → user_id + token).

Flow (identique a tous les clients Komoot connus) :
1. GET https://api.komoot.de/v006/account/email/{email}/
   avec Basic(email, password)
   -> renvoie {username: <user_id>, password: <api_token>, ...}
2. Toutes les requetes suivantes utilisent Basic(user_id, api_token).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .logging_setup import get_logger

_log = get_logger()

AUTH_URL = "https://api.komoot.de/v006/account/email"
SESSION_PATH = Path(os.path.expanduser("~/.config/komoot-mcp/session.json"))


class KomootAuth:
    """Gere l'authentification et la persistance de session Komoot."""

    def __init__(self) -> None:
        load_dotenv()
        self.email = os.environ.get("KOMOOT_EMAIL", "").strip()
        self.password = os.environ.get("KOMOOT_PASSWORD", "").strip()
        if not self.email or not self.password:
            raise RuntimeError(
                "KOMOOT_EMAIL / KOMOOT_PASSWORD manquants. "
                "Renseigne-les dans .env (voir .env.example)."
            )
        self._user_id: str | None = None
        self._token: str | None = None
        self._load_session()

    def _load_session(self) -> None:
        if SESSION_PATH.exists():
            try:
                data = json.loads(SESSION_PATH.read_text())
                self._user_id = data["user_id"]
                self._token = data["token"]
                _log.info("session_loaded", extra={"event": "session_loaded", "user_id": self._user_id})
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_session(self) -> None:
        SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        SESSION_PATH.write_text(json.dumps({
            "user_id": self._user_id,
            "token": self._token,
        }))
        try:
            os.chmod(SESSION_PATH, 0o600)
        except OSError:
            pass

    def _login(self) -> None:
        """Authentification initiale aupres de Komoot."""
        url = f"{AUTH_URL}/{self.email}/"
        resp = httpx.get(url, auth=(self.email, self.password), timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Komoot login echoue ({resp.status_code}): {resp.text[:500]}"
            )
        data = resp.json()
        self._user_id = str(data["username"])
        self._token = data["password"]
        self._save_session()
        _log.info("login_ok", extra={"event": "login_ok", "user_id": self._user_id})

    @property
    def user_id(self) -> str:
        if self._user_id is None:
            self._login()
        return self._user_id  # type: ignore[return-value]

    def auth_tuple(self) -> tuple[str, str]:
        """Retourne (user_id, token) pour httpx auth= parameter."""
        if self._user_id is None or self._token is None:
            self._login()
        return (self._user_id, self._token)  # type: ignore[return-value]

    def refresh(self) -> None:
        """Force une re-authentification (si le token a expire)."""
        _log.info("session_refresh", extra={"event": "session_refresh"})
        self._user_id = None
        self._token = None
        if SESSION_PATH.exists():
            SESSION_PATH.unlink()
        self._login()
