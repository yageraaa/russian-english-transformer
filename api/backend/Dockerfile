FROM python:3.10-slim
WORKDIR /app
COPY api/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
CMD ["uvicorn", "api.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]