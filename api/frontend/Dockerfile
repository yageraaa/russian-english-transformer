FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./api/frontend ./api/frontend
COPY ./api/.env ./api/.env
CMD ["streamlit", "run", "api/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]