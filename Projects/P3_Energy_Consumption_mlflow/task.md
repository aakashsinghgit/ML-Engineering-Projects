# Task Tracker

## Phase 1: Fix API & Dependencies
- [x] 1.1 — Delete dead code (lines 1–102) from api.py
- [x] 1.2 — Add `httpx>=0.24.0` to Requirements.txt
- [x] 1.3 — Verify `__init__.py` exports
- [x] 1.4 — Install httpx and run tests

## Phase 2: Run Locally (End-to-End)
- [x] 2.1 — Verify MLflow UI starts
- [x] 2.2 — Verify FastAPI server starts
- [x] 2.3 — Test endpoints
- [x] 2.4 — Test training run via API

## Phase 3: Dockerize
- [x] 3.1 — Create Dockerfile.serving
- [x] 3.2 — Create init-scripts/create-bucket.sh (Inline in docker-compose.yml)
- [x] 3.3 — Update docker-compose.yml
- [x] 3.4 — Create .env file
- [x] 3.5 — Update .gitignore and .dockerignore
- [x] 3.6 — Update mlflow_config.py for MLFLOW_TRACKING_URI
- [ ] 3.7 — Verify Docker stack (requires Docker daemon to be started by user)

## Phase 4: Harden
- [x] 4.1 — Create tests/conftest.py and pytest.ini
- [x] 4.2 — Expand test coverage in test_api.py
- [~] 4.3 — Create GitHub Actions CI workflow (Deferred for later)
- [x] 4.4 — Create readiness check script
