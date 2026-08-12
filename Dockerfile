FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app/src:/app/backend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY backend/ backend/
COPY models/ models/
COPY data/processed/ data/processed/
COPY data/features/ data/features/

EXPOSE 8001

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
