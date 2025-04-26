# FastAPI LangGraph Template Setup Guide

A comprehensive guide covering both **Makefile + Docker Compose** workflows (via WSL/Ubuntu) and **plain Docker** commands, plus common pitfalls encountered and how to resolve them.

---

## 1. Prerequisites

- **OS**: Windows + WSL (Ubuntu) or macOS/Linux
- **Docker** & **Docker Compose**
- **Python 3.13+** & `uv` CLI (`pip install uv`)
- **Make** (via Ubuntu in WSL) or local `make` on macOS/Linux

---

## 2. Clone & Environment

```bash
# Clone your fork
git clone <repository-url> fast-api-langgraph-fastmcp
cd fast-api-langgraph-fastmcp

# Copy example env to development
cp .env.example .env.development
# Then edit .env.development and set:
#   - JWT_SECRET_KEY
#   - LLM_API_KEY and/or OPENAI_API_KEY
#   - POSTGRES_URL (e.g. postgresql://postgres:mysecretpw@db:5432/fastapi-langgraph-mcp-dev)
```

Verify your `.env.development` includes:
```ini
APP_ENV=development
JWT_SECRET_KEY=<your-key>
LLM_API_KEY=<your-key>
OPENAI_API_KEY=<your-key>
POSTGRES_URL=postgresql://postgres:mysecretpw@db:5432/fastapi-langgraph-mcp-dev
```

---

## 3. Build & Run: Makefile + Docker Compose (WSL/Ubuntu)

> **Image**: `fast-api-langgraph-fastmcp_app`

1. **Build** API image:
   ```bash
   wsl -d Ubuntu 
  ```
  ```bash
  make docker-build ENV=development
   ```
2. **Start full stack** (API + Postgres + Prometheus + Grafana):
   ```bash
   make docker-compose-up ENV=development
   ```
3. **Verify running**:
   ```bash
   docker ps | grep fast-api-langgraph-fastmcp
   ```
4. **Endpoints**:
   - API health:     `http://localhost:8000/health`
   - Swagger UI:     `http://localhost:8000/docs`
   - Prometheus:     `http://localhost:9190`
   - Grafana:        `http://localhost:3300` (admin/admin)
5. **Logs / Stop / Down**:
   ```bash
   make docker-compose-logs ENV=development  # follow all logs
   make docker-compose-down ENV=development  # tear down stack
   ```


## 6. Development Workflow

- **Code changes** under `/app/app`: hot-reloaded by Uvicorn (`--reload`). No rebuild needed—just restart API container.
- **Dependencies or Dockerfile changes**: run `make docker-build ENV=development` to rebuild image.

---

## 7. CI / Staging / Production

Use the same Make targets with different `ENV` values:

| Environment | Build Command                         | Deployment                                                      |
|-------------|---------------------------------------|-----------------------------------------------------------------|
| **Test**    | `make docker-build ENV=test`      | CI: tag and push `fast-api-langgraph-fastmcp_app:test`         |
| **Stage**   | `make docker-build ENV=staging`   | `docker run -p 80:8000 … fast-api-langgraph-fastmcp_app:staging`|
| **Prod**    | `make docker-build ENV=production`| Orchestrator (K8s/Compose): `:production`                      |

> Secrets are injected via `--build-arg` and/or runtime `-e` flags, managed by `docker-entrypoint.sh`.

---

## 8. Quick Reference Cheat Sheet

```bash
# Build & run full stack (WSL)
wsl -d Ubuntu -- cd … && make docker-build-env ENV=development && make docker-compose-up ENV=development

# Plain Docker build & run
docker build --build-arg APP_ENV=development -t fast-api-langgraph-dev .
docker run -d --name fastapi-langgraph-dev -p 8000:8000 … fast-api-langgraph-dev

# Logs & stop
make docker-compose-logs ENV=development
make docker-compose-down ENV=development
docker logs fastapi-langgraph-dev -f
docker stop fastapi-langgraph-dev && docker rm fastapi-langgraph-dev

# Health check
curl http://localhost:8000/health
```

