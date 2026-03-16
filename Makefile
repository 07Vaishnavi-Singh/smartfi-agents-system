.PHONY: up down logs status

# Start all services: infrastructure (Docker) + FastAPI server
up:
	docker compose up -d
	@echo "Waiting for Redis and Qdrant to be ready..."
	@sleep 2
	@echo "Starting FastAPI server (PID will be saved to .fastapi.pid)..."
	PYTHONPATH=src nohup uv run uvicorn investment_research_system.api.app:create_app \
		--factory --reload --port 8000 > .fastapi.log 2>&1 & echo $$! > .fastapi.pid
	@echo ""
	@echo "All services up:"
	@echo "  Redis:   localhost:6379"
	@echo "  Qdrant:  localhost:6333"
	@echo "  FastAPI: localhost:8000"
	@echo "  Docs:    localhost:8000/docs"
	@echo ""
	@echo "FastAPI logs: tail -f .fastapi.log"

# Stop everything: FastAPI server + Docker containers
down:
	@if [ -f .fastapi.pid ]; then \
		echo "Stopping FastAPI server (PID $$(cat .fastapi.pid))..."; \
		kill $$(cat .fastapi.pid) 2>/dev/null || true; \
		rm -f .fastapi.pid; \
	else \
		echo "No FastAPI PID file found, skipping..."; \
	fi
	docker compose down
	@echo "All services stopped."

# Tail FastAPI logs
logs:
	@tail -f .fastapi.log

# Check status of all services
status:
	@echo "=== Docker ==="
	@docker compose ps
	@echo ""
	@echo "=== FastAPI ==="
	@if [ -f .fastapi.pid ] && kill -0 $$(cat .fastapi.pid) 2>/dev/null; then \
		echo "Running (PID $$(cat .fastapi.pid))"; \
	else \
		echo "Not running"; \
	fi
