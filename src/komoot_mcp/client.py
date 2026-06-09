"""Komoot REST client — async httpx wrapper with auto-reauth."""

from __future__ import annotations

import time
from typing import Any

import httpx

from .auth import KomootAuth
from .logging_setup import get_logger

_log = get_logger()

API_BASE = "https://api.komoot.de/v007"
WWW_BASE = "https://www.komoot.com/api"


class KomootError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Komoot API {status}: {body}")
        self.status = status
        self.body = body


class KomootClient:
    def __init__(self, auth: KomootAuth | None = None) -> None:
        self._auth = auth or KomootAuth()
        self._client = httpx.AsyncClient(timeout=60)

    @property
    def user_id(self) -> str:
        return self._auth.user_id

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        data: Any = None,
        files: dict | None = None,
        headers: dict | None = None,
        raw: bool = False,
    ) -> Any:
        """Effectue une requete HTTP avec Basic Auth et retry sur 401."""
        for attempt in range(2):
            t0 = time.time()
            resp = await self._client.request(
                method,
                url,
                params=params,
                json=json_body,
                data=data,
                files=files,
                headers=headers,
                auth=self._auth.auth_tuple(),
            )
            dt_ms = int((time.time() - t0) * 1000)
            # Extraire un path court pour le log
            short_path = url.replace(API_BASE, "").replace(WWW_BASE, "")
            _log.info(
                "http",
                extra={
                    "event": "http",
                    "method": method,
                    "path": short_path,
                    "status": resp.status_code,
                    "duration_ms": dt_ms,
                    "attempt": attempt,
                },
            )
            if resp.status_code == 401 and attempt == 0:
                _log.info("reauth", extra={"event": "reauth", "reason": "401 received"})
                self._auth.refresh()
                continue
            if resp.status_code >= 400:
                body = resp.text[:800]
                _log.error(
                    "http_error",
                    extra={
                        "event": "http_error",
                        "method": method,
                        "path": short_path,
                        "status": resp.status_code,
                        "body": body,
                    },
                )
                raise KomootError(resp.status_code, body)
            if raw:
                return resp.content
            if resp.status_code == 204 or not resp.content:
                return None
            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                return resp.json()
            return resp.content
        raise KomootError(401, "Authentication failed after retry.")

    # ── Convenience wrappers ─────────────────────────────────────────────

    async def get(self, path: str, raw: bool = False, **params) -> Any:
        url = f"{API_BASE}{path}" if path.startswith("/") else path
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("GET", url, params=clean, raw=raw)

    async def get_www(self, path: str, raw: bool = False, **params) -> Any:
        """Requete via www.komoot.com/api (pour routing/import)."""
        url = f"{WWW_BASE}{path}" if path.startswith("/") else path
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("GET", url, params=clean, raw=raw)

    async def post(self, path: str, *, json_body: dict | None = None,
                   data: Any = None, files: dict | None = None,
                   headers: dict | None = None) -> Any:
        url = f"{API_BASE}{path}" if path.startswith("/") else path
        return await self.request("POST", url, json_body=json_body, data=data,
                                  files=files, headers=headers)

    async def post_www(self, path: str, *, json_body: dict | None = None,
                       data: Any = None, files: dict | None = None,
                       headers: dict | None = None, params: dict | None = None) -> Any:
        url = f"{WWW_BASE}{path}" if path.startswith("/") else path
        return await self.request("POST", url, json_body=json_body, data=data,
                                  files=files, headers=headers, params=params)

    async def patch(self, path: str, **fields) -> Any:
        url = f"{API_BASE}{path}" if path.startswith("/") else path
        return await self.request("PATCH", url, json_body={k: v for k, v in fields.items() if v is not None})

    async def delete(self, path: str) -> Any:
        url = f"{API_BASE}{path}" if path.startswith("/") else path
        return await self.request("DELETE", url)
