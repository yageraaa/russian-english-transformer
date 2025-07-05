FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./api/backend ./api/backend
COPY ./api/ml ./api/ml
COPY ./tokenizer ./tokenizer
COPY ./models ./models
COPY ./api/.env ./api/.env
CMD ["uvicorn", "api.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]