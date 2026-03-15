.PHONY: dev down build logs logs-api logs-inference logs-manager logs-worker \
       infra infra-down web web-install clean restart status ps

# ── Full Stack ────────────────────────────────────────────────
dev: ## Start all backend services + infrastructure
	docker compose up --build -d

down: ## Stop everything
	docker compose down

restart: ## Restart all services
	docker compose down
	docker compose up --build -d

build: ## Rebuild the app image without starting
	docker compose build

# ── Infrastructure Only ───────────────────────────────────────
infra: ## Start only infra (qdrant, redpanda, redis)
	docker compose up -d qdrant redpanda redis

infra-down: ## Stop only infra
	docker compose stop qdrant redpanda redis

# ── Frontend ──────────────────────────────────────────────────
web-install: ## Install frontend dependencies
	cd web && npm install

web: ## Start the React dev server
	cd web && npm run dev

web-build: ## Build the frontend for production
	cd web && npm run build

web-lint: ## Lint the frontend
	cd web && npm run lint

# ── Logs ──────────────────────────────────────────────────────
logs: ## Tail all service logs
	docker compose logs -f

logs-api: ## Tail API logs
	docker compose logs -f api

logs-inference: ## Tail inference server logs
	docker compose logs -f inference

logs-manager: ## Tail manager logs
	docker compose logs -f manager

logs-worker: ## Tail worker logs
	docker compose logs -f worker

# ── Status ────────────────────────────────────────────────────
ps: ## Show running containers
	docker compose ps

status: ## Show container status + ports
	@echo "=== Containers ==="
	@docker compose ps
	@echo ""
	@echo "=== Ports ==="
	@echo "  API:        http://localhost:8000"
	@echo "  Inference:  http://localhost:8001"
	@echo "  Qdrant:     http://localhost:6333"
	@echo "  Redpanda:   http://localhost:9094"
	@echo "  Redis:      localhost:6379"
	@echo "  Web (dev):  http://localhost:5173"

# ── Cleanup ───────────────────────────────────────────────────
clean: ## Stop everything and remove volumes
	docker compose down -v

# ── Help ──────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
