FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# duckdb_service.py LOADs the spatial extension on every connection — baked
# into the image at build time so containers don't need network access (or a
# download delay) on every startup.
RUN python -c "import duckdb; duckdb.connect().install_extension('spatial')"

# Overwritten by a bind mount in docker-compose.yml for dev (hot reload via
# --reload below); this COPY just makes the image runnable standalone too.
COPY app ./app

EXPOSE 8000

# --reload watches /app/app for changes — paired with docker-compose.yml
# mounting the host app/backend/app directory there, edits on the host restart
# the server automatically without rebuilding the image.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
