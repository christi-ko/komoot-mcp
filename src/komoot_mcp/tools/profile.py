"""MCP tools for Komoot user profile."""

from __future__ import annotations

from typing import Any

from ..client import KomootClient


def register(mcp, client: KomootClient) -> None:
    @mcp.tool()
    async def get_user_profile() -> dict[str, Any]:
        """Profil de l'utilisateur Komoot connecte (nom, avatar, stats)."""
        return await client.get(f"/users/{client.user_id}")
