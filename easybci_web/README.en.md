# EasyBCI Agent — Web UI

Three-column workspace for interacting with EasyBCI Agent: manage sessions, chat with the agent via SSE streaming, and browse source/output files — all in a single view.

📖 **Language / 语言:** [中文](README.md) · **English**

## Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 19 |
| Build | Vite 7 |
| Styling | Tailwind CSS 4 (`@tailwindcss/vite` plugin) |
| State | zustand 5 (session / ui / toast / theme stores) |
| Markdown | react-markdown 10 + remark-gfm 4 |
| Syntax | shiki 3 (WASM, on-demand grammar loading) |
| Virtualization | @tanstack/react-virtual 3 |
| Type checking | TypeScript 5.9 (`strict: true`) |

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Browser (localhost:5173 dev / served from web_dist in production) │
│                                                                    │
│  ┌────────────┐    ┌──────────────────┐    ┌───────────────┐       │
│  │SessionPanel│    │ConversationPanel  │    │WorkspacePanel │      │
│  │  sessions  │    │  messages + chat  │    │  file trees   │      │
│  └─────┬──────┘    └────────┬─────────┘    └───────┬───────┘       │
│        │                    │                      │               │
│  useSessionList      useConversation          useWorkspace         │
│        │                    │                      │               │
└────────┼────────────────────┼──────────────────────┼───────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
   Dashboard REST       Gateway SSE            Dashboard REST
   port 9119            port 8642              port 9119
   /api/sessions/*      /v1/runs/*             /api/files/*
```

**Data flow**: User message → `startRun()` POST to Gateway → SSE stream pushes events → `useConversation` updates messages → `useWorkspace` extracts file paths from tool events → right panel refreshes file tree.

## Development

The WebUI depends on two backend services. The simplest path is a single command — `easybci dashboard` auto-starts the Gateway and serves the built WebUI; run Vite separately only when you need frontend hot-reload (HMR):

```bash
# One command: auto-starts Gateway + serves the production build + opens the browser
easybci dashboard          # → http://localhost:9119
```

> **Port auto-recovery**: re-running `easybci dashboard` automatically terminates any stale/hung previous process and reuses the port; if a live old instance exists, the new one slides to 9120/9121 and shares the same Gateway. `--port 9119` is strict mode — errors on conflict.

```bash
# ── Frontend hot-reload development (three terminals) ──
# Terminal 1: Gateway API (agent execution + OpenAI-compatible HTTP/SSE)
API_SERVER_ENABLED=true API_SERVER_PORT=8642 python -m services.gateway.run

# Terminal 2: Dashboard REST (sessions, config, file browser) — no auto browser, no Gateway auto-start
easybci dashboard --port 9119 --no-open --no-gateway

# Terminal 3: Frontend dev server (HMR + proxy)
cd easybci_web
npm install
npm run dev                # → http://localhost:5173
```

The Vite dev server proxies:
- `/api/*` → `http://127.0.0.1:9119` (Dashboard)
- `/v1/*` → `http://127.0.0.1:8642` (Gateway)

Proxy targets are configurable via env vars `EASYBCI_GATEWAY_URL` and `EASYBCI_DASHBOARD_URL`.

## Production Build

```bash
cd easybci_web
npm run build
```

Outputs to `../easybci_cli/web_dist/`. The Dashboard server serves these as a static SPA. Built assets are included in the Python package via `pyproject.toml` package-data.

## Project Structure

```
easybci_web/src/
├── App.tsx / main.tsx / index.css   # Wiring + ErrorBoundary entry + CSS variables (light/dark)
├── layouts/                         # Responsive 3-column grid + resize handles + overlays
├── panels/                          # SessionPanel / ConversationPanel / WorkspacePanel — main columns
├── components/                      # 24 presentation components: MessageBubble / Pipeline cards /
│                                    # QCCard / dialogs / banners / Toast / ErrorBoundary / etc.
├── hooks/                           # useConversation (SSE stream) / useWorkspace (path extraction) /
│                                    # useSessionList / useConnectionStatus / useHighlighter / etc.
├── stores/                          # zustand: sessionStore / uiStore / themeStore / toastStore
└── lib/                             # api.ts (Dashboard REST) / runsClient.ts (Gateway SSE) /
                                     # offlineCache.ts / pipelineParse.ts / utilities
```

Specific filenames are visible in the IDE; the list above shows group responsibilities only.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_SERVER_KEY` | _(empty)_ | Gateway auth token (triggers fetch fallback for SSE) |
| `EASYBCI_GATEWAY_URL` | `http://127.0.0.1:8642` | Gateway proxy target |
| `EASYBCI_DASHBOARD_URL` | `http://127.0.0.1:9119` | Dashboard proxy target |

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Vite dev server with HMR (port 5173) |
| `npm run build` | TypeScript check (`tsc -b`) + Vite production build |
| `npm run lint` | ESLint (react-hooks + react-refresh rules) |
| `npm run preview` | Preview production build locally |

## TypeScript Configuration

- `strict: true`, `noUnusedLocals: true`, `noUnusedParameters: true`
- Path alias: `@/` → `./src/*`
- Target: ES2023, module: ESNext, moduleResolution: bundler
