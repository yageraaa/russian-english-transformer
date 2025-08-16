import subprocess
import sys
import os

def start_mlflow_ui():
    try:
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui", 
            "--backend-store-uri", "file:./mlruns",
            "--host", "0.0.0.0",
            "--port", "5000"
        ])
    except KeyboardInterrupt:
        print("\nMLflow UI остановлен")
    except Exception as e:
        print(f"Ошибка при запуске MLflow UI: {e}")

if __name__ == "__main__":
    start_mlflow_ui() 