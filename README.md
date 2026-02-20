# What?

Semantic Cuts, a project born out of my curiosity for multimodal semantic search, a multimodal video search engine — you give it a natural language query like "a red car" or "people laughing" and it finds the exact moments in videos matching that description. It currently uses OpenAI's CLIP model to bridge text and video frames via shared embeddings. Efficiency is a feature — not a trade-off.

# Demo

[![Watch the video on YT](https://img.youtube.com/vi/-VDlq39URtY/maxresdefault.jpg)](https://youtu.be/-VDlq39URtY)

# Architecture

```
                         ┌──────────────┐
                         │  React UI    │ :5173
                         └──────┬───────┘
                                │
                   ┌────────────┴────────────┐
                   │                         │
            ┌──────┴───────┐         ┌───────┴──────┐
            │   API Server │ :8000   │  Inference   │ :8001
            │  (main.py)   │         │ (server.py)  │
            └──────┬───────┘         └───────▲──────┘
                   │                         │
                   │ Kafka                   │ /embed
                   ▼                         │
            ┌──────────────┐         ┌───────┴──────┐
            │   Manager    │ ──────► │    Worker     │
            │ (manager.py) │  Kafka  │  (minion.py)  │
            └──────────────┘         └──────────────┘
                   │                         │
                   ▼                         ▼
            ┌──────────────┐         ┌──────────────┐
            │    Redis     │         │    Qdrant    │
            └──────────────┘         └──────────────┘
```

# Running

## With Docker / Podman

```bash
docker-compose up --build
# or
podman-compose up --build # Just trying out Podman for its rootless

# Restart all the containers:
docker-compose down && docker-compose up --build
# or
podman-compose down && podman-compose up --build
```

This starts all 7 services:

| Service | Port | Description |
|---------|------|-------------|
| Inference | 8001 | CLIP model, embedding + search |
| API | 8000 | REST API for the frontend |
| Manager | — | Video download + chunk dispatch |
| Worker | — | Scene detection + embedding pipeline |
| Qdrant | 6333 | Vector database |
| Redpanda | 9094 | Kafka-compatible message broker |
| Redis | 6379 | Job progress tracking |

## Local (incase you want without containers)

Start infrastructure first (Qdrant, Redpanda, Redis), then in separate terminals:

```bash
python3 -m app.server    # Inference engine on :8001
python3 -m app.main      # API server on :8000
python3 -m app.manager   # Job orchestrator
python3 -m app.minion    # Worker process
```

For the React frontend:

```bash
cd web && npm install && npm run dev
```
