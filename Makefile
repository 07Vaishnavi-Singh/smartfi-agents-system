.PHONY: up down logs logs-api logs-ui status

# Start all services: infrastructure (Docker) + FastAPI + Streamlit
up:
	docker compose up -d
	@echo "Waiting for Redis and Qdrant to be ready..."
	@sleep 2
	@echo "Starting FastAPI server..."
	PYTHONPATH=src nohup uv run uvicorn investment_research_system.api.app:create_app \
		--factory --reload --port 8000 > .fastapi.log 2>&1 & echo $$! > .fastapi.pid
	@echo "Starting Streamlit frontend..."
	nohup uv run streamlit run src/investment_research_system/ui/app.py \
		--server.port 8501 --server.headless true > .streamlit.log 2>&1 & echo $$! > .streamlit.pid
	@echo ""
	@echo "All services up:"
	@echo "  Redis:     localhost:6379"
	@echo "  Qdrant:    localhost:6333"
	@echo "  FastAPI:   localhost:8000"
	@echo "  API Docs:  localhost:8000/docs"
	@echo "  Streamlit: localhost:8501"
	@echo ""
	@echo "Logs: make logs-api | make logs-ui"

# Stop everything: Streamlit + FastAPI + Docker containers
down:
	@if [ -f .streamlit.pid ]; then \
		echo "Stopping Streamlit (PID $$(cat .streamlit.pid))..."; \
		kill $$(cat .streamlit.pid) 2>/dev/null || true; \
		rm -f .streamlit.pid; \
	else \
		echo "No Streamlit PID file found, skipping..."; \
	fi
	@if [ -f .fastapi.pid ]; then \
		echo "Stopping FastAPI (PID $$(cat .fastapi.pid))..."; \
		kill $$(cat .fastapi.pid) 2>/dev/null || true; \
		rm -f .fastapi.pid; \
	else \
		echo "No FastAPI PID file found, skipping..."; \
	fi
	@# Safety net: kill anything still holding the ports
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8501 | xargs kill -9 2>/dev/null || true
	docker compose down
	@echo "All services stopped."

# Tail logs
logs-api:
	@tail -f .fastapi.log

logs-ui:
	@tail -f .streamlit.log

logs:
	@tail -f .fastapi.log .streamlit.log

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
	@echo ""
	@echo "=== Streamlit ==="
	@if [ -f .streamlit.pid ] && kill -0 $$(cat .streamlit.pid) 2>/dev/null; then \
		echo "Running (PID $$(cat .streamlit.pid))"; \
	else \
		echo "Not running"; \
	fi
