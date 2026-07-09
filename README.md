# firewall-for-ai-workers

A Cloudflare Workers demo that hosts a multimodal AI chat agent with safety guardrails, tool calling, scheduling, MCP server support, and a modern React UI.

## Overview

This project runs on the Cloudflare Workers platform and combines the following:

- **Agents SDK + `@cloudflare/ai-chat`** for Durable Object-backed chat state
- **Remote `llamacpp` model endpoint** routed through the Cloudflare AI Gateway
- **Workers AI safety guardrails** using `@cf/meta/llama-3.2-1b-instruct`
- **Tool-enabled assistant** with weather, timezone, math, scheduling, and image generation
- **MCP server support** for connecting external tool servers with OAuth callbacks
- **React 19 + Vite + Tailwind CSS 4** UI with theme switching and toast notifications
- **R2 bucket** for storing generated images

## Features

- Cloudflare Workers agent chat using `@cloudflare/ai-chat` and the `agents` SDK
- Server-side streaming response using the Vercel AI SDK (`ai`)
- Safety guardrails that analyze incoming user messages for harmful content
- Built-in tools: weather, timezone, calculator, task scheduling, image generation
- User approval flow for sensitive tool calls (e.g. large calculations)
- MCP server management with add/remove via RPC and OAuth support
- Scheduled task execution with broadcast notifications to connected clients
- File attachments and image preview in the chat UI
- Dark/light theme toggle and toast notifications
- Durable Object-backed persistence for chat messages and schedules

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

- `CLOUDFLARE_ACCOUNT_ID` — Cloudflare account ID
- `CLOUDFLARE_API_TOKEN` — Cloudflare API token
- `LLAMA_CPP_URL` — Base URL of the remote Llama model endpoint
- `LLAMA_CPP_KEY` — API key for the remote model endpoint
- `LLAMA_CPP_MODEL` — Model name to use for chat completion

These are wired through `env.d.ts` and consumed in `src/server.ts`.

## Key Files

- `src/server.ts` — Durable Object chat agent logic, tool definitions, model integration, scheduling, and safety guardrails
- `src/app.tsx` — React chat UI with attachments, approvals, MCP server management, and theme controls
- `wrangler.jsonc` — Cloudflare Worker configuration, Durable Objects, AI binding, R2 bucket, and asset routing
- `env.d.ts` — typed Cloudflare environment bindings
- `vite.config.ts` — Vite build configuration with the Cloudflare Workers plugin

## Notes

- The worker uses the `AI` binding with `remote: true` and routes generation through Cloudflare AI Gateway.
- `wrangler.jsonc` configures `assets` to serve the SPA and routes `/agents/*` and `/oauth/*` through the Worker.
- Generated images are stored in the configured R2 bucket and served via `R2_PUBLIC_URL`.
- Use `npm run check` to run formatting, linting, and TypeScript checks.

## Scripts

- `npm run dev` — run local development server
- `npm run start` — alias for `vite dev`
- `npm run deploy` — build and deploy to Cloudflare
- `npm run types` — generate Wrangler environment types
- `npm run format` — format code with `oxfmt`
- `npm run lint` — lint `src/` with `oxlint`
- `npm run check` — format check, lint, and TypeScript validation
