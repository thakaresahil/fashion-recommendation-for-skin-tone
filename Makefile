.PHONY: install install-ml images api web test lint format embeddings docker

install:            ## Install backend and frontend dependencies
	pip install -r backend/requirements-dev.txt
	cd frontend && npm ci

install-ml:         ## Add optional CLIP dependencies for dense search
	pip install -r backend/requirements-ml.txt

images:             ## Download product images (SAMPLE=1500 for a small subset)
	cd backend && python -m scripts.fetch_images $(if $(SAMPLE),--sample $(SAMPLE))

api:                ## Run the API with auto-reload on :8000
	cd backend && uvicorn app.main:app --reload --port 8000

web:                ## Run the Vite dev server on :5173 (proxies /api to :8000)
	cd frontend && npm run dev

test:               ## Run backend tests and frontend type-check
	cd backend && python -m pytest -q
	cd frontend && npm run typecheck

lint:
	cd backend && ruff check . && ruff format --check .

format:
	cd backend && ruff format . && ruff check --fix .

embeddings:         ## Build CLIP embeddings (needs install-ml; uses data/images if present)
	cd backend && python -m scripts.build_embeddings

docker:             ## Build and run the production container on :8000
	docker compose up --build
