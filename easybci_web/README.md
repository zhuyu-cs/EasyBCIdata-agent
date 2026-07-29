# EasyBCI Agent — Web UI

与 EasyBCI Agent 交互的三栏工作台：管理会话、通过 SSE 流式与智能体对话、浏览源数据与输出文件——全部在单页内完成。

📖 **语言 / Language:** **中文** · [English](README.en.md)

## 技术栈

| 层 | 技术 |
|----|------|
| 框架 | React 19 |
| 构建 | Vite 7 |
| 样式 | Tailwind CSS 4（`@tailwindcss/vite` 插件） |
| 状态 | zustand 5（session / ui / toast / theme stores） |
| Markdown | react-markdown 10 + remark-gfm 4 |
| 语法高亮 | shiki 3（WASM，按需加载 grammar） |
| 虚拟滚动 | @tanstack/react-virtual 3 |
| 类型检查 | TypeScript 5.9（`strict: true`） |

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│  浏览器（开发 localhost:5173 / 生产由 web_dist 静态托管）            │
│                                                                 │
│  ┌────────────┐    ┌──────────────────┐    ┌───────────────┐    │
│  │SessionPanel│    │ConversationPanel │    │WorkspacePanel │    │
│  │   会话列表   │   │  消息流 + 输入框    │   │   文件树       │    │
│  └─────┬──────┘    └────────┬─────────┘    └───────┬───────┘    │
│        │                    │                      │            │
│  useSessionList      useConversation          useWorkspace      │
│        │                    │                      │            │
└────────┼────────────────────┼──────────────────────┼────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
   Dashboard REST       Gateway SSE            Dashboard REST
   端口 9119            端口 8642              端口 9119
   /api/sessions/*      /v1/runs/*             /api/files/*
```

**数据流**：用户发送消息 → `startRun()` POST 到 Gateway → SSE 推送事件流 → `useConversation` 更新消息 → `useWorkspace` 从工具事件中提取文件路径 → 右侧面板刷新文件树。

## 开发

WebUI 依赖两个后端服务。最简单的方式是一条命令——`easybci dashboard` 会自动拉起 Gateway 并托管已构建的 WebUI；只有需要前端热更新（HMR）时才单独跑 Vite：

```bash
# 一条命令：自动启动 Gateway + 托管生产构建 + 打开浏览器
easybci dashboard          # → http://localhost:9119
```

> **端口自动恢复**：重复运行 `easybci dashboard` 会自动 terminate 上一次残留的卡死进程并复用端口；如果有正在运行的旧实例，新实例会滑到 9120/9121 并共享同一个 Gateway。`--port 9119` 是严格模式，遇冲突直接报错。

```bash
# ── 前端热更新开发（三个终端）──
# 终端 1：Gateway API（Agent 执行 + OpenAI 兼容 HTTP/SSE）
API_SERVER_ENABLED=true API_SERVER_PORT=8642 python -m services.gateway.run

# 终端 2：Dashboard REST（会话、配置、文件浏览），关闭自动开浏览器与 Gateway 自启
easybci dashboard --port 9119 --no-open --no-gateway

# 终端 3：前端开发服务器（HMR + 代理）
cd easybci_web
npm install
npm run dev                # → http://localhost:5173
```

Vite 开发服务器代理：
- `/api/*` → `http://127.0.0.1:9119`（Dashboard）
- `/v1/*` → `http://127.0.0.1:8642`（Gateway）

代理目标可通过环境变量 `EASYBCI_GATEWAY_URL` 与 `EASYBCI_DASHBOARD_URL` 配置。

## 生产构建

```bash
cd easybci_web
npm run build
```

输出到 `../easybci_cli/web_dist/`，由 Dashboard 服务器作为静态 SPA 托管；构建产物通过 `pyproject.toml` 的 package-data 打进 Python 包。

## 目录结构

```
easybci_web/src/
├── App.tsx / main.tsx / index.css   # 装配 + ErrorBoundary 入口 + CSS 变量（light/dark）
├── layouts/                         # 响应式三栏栅格 + 拖拽手柄 + overlay
├── panels/                          # SessionPanel / ConversationPanel / WorkspacePanel — 三栏主体
├── components/                      # 24 个展示组件：MessageBubble / Pipeline 卡 / QCCard /
│                                    # 对话框 / 横幅 / Toast / ErrorBoundary 等
├── hooks/                           # useConversation（SSE 流）/ useWorkspace（路径提取）/
│                                    # useSessionList / useConnectionStatus / useHighlighter / 等
├── stores/                          # zustand：sessionStore / uiStore / themeStore / toastStore
└── lib/                             # api.ts（Dashboard REST）/ runsClient.ts（Gateway SSE）/
                                     # offlineCache.ts / pipelineParse.ts / 工具函数
```

具体文件名 IDE 中可见；上面只列分组职责。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VITE_API_SERVER_KEY` | _(空)_ | Gateway 鉴权 token（触发 SSE 的 fetch 回退路径） |
| `EASYBCI_GATEWAY_URL` | `http://127.0.0.1:8642` | Gateway 代理目标 |
| `EASYBCI_DASHBOARD_URL` | `http://127.0.0.1:9119` | Dashboard 代理目标 |

## 脚本

| 命令 | 说明 |
|------|------|
| `npm run dev` | Vite 开发服务器 + HMR（端口 5173） |
| `npm run build` | TypeScript 检查（`tsc -b`）+ Vite 生产构建 |
| `npm run lint` | ESLint（react-hooks + react-refresh 规则） |
| `npm run preview` | 本地预览生产构建 |

## TypeScript 配置

- `strict: true`、`noUnusedLocals: true`、`noUnusedParameters: true`
- 路径别名：`@/` → `./src/*`
- target：ES2023，module：ESNext，moduleResolution：bundler
