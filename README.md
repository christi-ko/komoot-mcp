# komoot-mcp

MCP server for [Komoot](https://www.komoot.com/) — the most comprehensive Komoot API wrapper available. Tours, highlights, route planning, exports, and more.

Built with Python + [FastMCP](https://github.com/jlowin/fastmcp). Combines capabilities from [kompy](https://github.com/Tsadoq/kompy), [KomootGPX](https://github.com/timschneeb/KomootGPX), [komPYoot](https://pypi.org/project/komPYoot/), and [export-komoot](https://github.com/pieterclaerhout/export-komoot) into a single MCP server.

## Features

### Tours
- **List tours** — filter by type (planned/recorded), sport, status, name, with sorting and pagination
- **Get tour detail** — distance, elevation, duration, sport, difficulty, surfaces
- **Update tour** — rename, change sport type, change visibility
- **Delete tour**

### Exports
- **Download GPX** — full GPS trace
- **Download FIT** — Garmin/ANT+ format

### Upload
- **Upload tour** — import GPX as recorded activity

### Highlights
- **Get highlight** — community points of interest
- **Get highlight tips** — community tips for a highlight
- **Get tour images** — photos attached to a tour

### Route planning
- **Import GPX route** — import and match to Komoot routing network
- **Plan route** — create a route from waypoints (lat/lng)
- **Create planned tour** — save a planned route as a Komoot tour

### Streams
- **Coordinates** — lat, lng, altitude, timestamp sequence
- **Surfaces** — surface types per segment (asphalt, gravel, dirt...)
- **Way types** — road, bike path, trail, etc.
- **Directions** — turn-by-turn navigation

### Profile
- **User profile** — name, avatar, stats, settings

### 18 tools total

## Setup

### 1. Configure credentials

```bash
cd komoot-mcp
cp .env.example .env
# Edit .env with your Komoot email and password
```

```env
KOMOOT_EMAIL=your.email@example.com
KOMOOT_PASSWORD=your_komoot_password
```

### 2. Install and run

```bash
uv sync
uv run python -m komoot_mcp.server
```

### 3. Add to Claude Desktop

```json
{
  "mcpServers": {
    "komoot": {
      "command": "uv",
      "args": ["--directory", "/path/to/komoot-mcp", "run", "python", "-m", "komoot_mcp.server"]
    }
  }
}
```

## Authentication

Basic Auth via Komoot's internal API (`/v006/account/email/{email}/`). Session token persisted in `~/.config/komoot-mcp/session.json` with automatic re-auth on 401.

> **Note:** Komoot does not have an official public API. This server uses the same undocumented v007 REST API that the web frontend and all major third-party tools use. It could break if Komoot changes their internal API.

## Logging

JSON-lines logs in `~/.config/komoot-mcp/logs/komoot-mcp.log`. Set `KOMOOT_MCP_LOG_LEVEL=DEBUG` in `.env`.

## License

MIT
