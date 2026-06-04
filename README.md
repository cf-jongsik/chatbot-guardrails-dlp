# Chatbot Demo

A Cloudflare Workers starter app for building an AI chat agent with tools, scheduling, and a modern React UI.

## Overview

This demo combines Cloudflare Agents, AI Chat, and a durable object-powered Worker backend to provide a chat experience that can:

- answer questions using a remote Llama model endpoint
- invoke tool calls for weather, timezone, math, and scheduling
- require user approval for sensitive tool actions
- connect to additional MCP tool servers
- schedule tasks for later execution
- persist conversation state with Durable Objects
- display notifications and theme switching in the browser UI

## Features

- Cloudflare Workers agent chat using `@cloudflare/ai-chat` and the `agents` SDK
- Tool-enabled assistant with server-side and client-side tool support
- File attachments and image handling in the UI
- Task scheduling and cancellation via `agents/schedule`
- Dark/light theme toggle and interactive toast notifications
- Durable Object-backed chat state and scheduled task execution

## Quick Start

1. Install dependencies:

```bash
npm install
```

2. Start local development:

```bash
npm run dev
```

3. Deploy to Cloudflare:

```bash
npm run deploy
```

4. Generate Wrangler types after changing bindings:

```bash
npm run types
```

## Environment Variables

The project expects the following environment variables at runtime:

- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`
- `LLAMA_CPP_URL`
- `LLAMA_CPP_KEY`
- `LLAMA_CPP_MODEL`

These are wired via `env.d.ts` and used by `src/server.ts`.

## Key Files

- `src/server.ts` — Durable Object chat agent logic, tool definitions, model integration, and schedule execution
- `src/app.tsx` — React chat UI with attachments, approvals, MCP server management, and theme controls
- `wrangler.jsonc` — Cloudflare Worker configuration, Durable Objects, AI binding, and asset routing
- `env.d.ts` — typed Cloudflare environment bindings

## Notes

- The Worker uses the `AI` binding with remote AI gateway support.
- `wrangler.jsonc` configures `assets` to serve the SPA and route `/agents/*` and `/oauth/*` through the Worker.
- Use `npm run check` to run formatting, linting, and TypeScript checks.

## Scripts

- `npm run dev` — run local development server
- `npm run start` — alias for `vite dev`
- `npm run deploy` — build and deploy to Cloudflare
- `npm run types` — generate Wrangler environment types
- `npm run format` — format code with `oxfmt`
- `npm run lint` — lint `src/` with `oxlint`
- `npm run check` — format check, lint, and TypeScript validation
