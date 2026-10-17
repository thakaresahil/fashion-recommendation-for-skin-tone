# ---- Build the React frontend ----
FROM node:22-alpine AS web
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- API + static frontend ----
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Set INSTALL_ML=true to bake in sentence-transformers for dense CLIP search (much larger image).
ARG INSTALL_ML=false
COPY backend/requirements*.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && if [ "$INSTALL_ML" = "true" ]; then pip install --no-cache-dir -r backend/requirements-ml.txt; fi

COPY backend/ backend/
COPY data/*.csv data/
COPY --from=web /app/frontend/dist frontend/dist

RUN useradd --create-home appuser
USER appuser
WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
