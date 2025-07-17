# Russian to English Translator

**Author**: German Berezin

This project provides a web-based Russian-to-English translation application using a transformer model, deployed via Docker with a Streamlit frontend, FastAPI backend, PostgreSQL database, and MinIO storage. It also includes scripts for training the transformer model.

## Prerequisites

- **Docker** and **Docker Compose**: Required for running the Streamlit application.
  ```bash
  sudo apt update
  sudo apt install docker.io docker-compose
  ```
- **Miniconda or Anaconda**: Required for training the model.
  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  ```
  Follow the installation prompts and initialize Conda.
- **Git**: To clone the repository.
  ```bash
  sudo apt install git
  ```
- **MinIO Client (mc)**: Optional, for interacting with MinIO storage.
  ```bash
  wget https://dl.min.io/client/mc/release/linux-amd64/mc
  chmod +x mc
  sudo mv mc /usr/local/bin/
  ```

## Project Structure

```
russian-english-transformer/
├── api/
│   ├── backend/
│   │   ├── database.py
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── requirements.txt
│   │   ├── s3_client.py
│   │   ├── settings.py
│   │   └── utils.py
│   ├── frontend/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── ml/
│   │   └── translator.py
│   ├── models/
│       └── configs/
│           └── config.yaml
├── checkpoints/
│   └── transformer_00.pt
├── docker-compose.yml
├── models/
│   ├── configs/
│   │   └── config.yaml
│   ├── core/
│   │   ├── embeddings.py
│   │   ├── feed_forward.py
│   │   ├── layer_norm.py
│   │   ├── linear_layer.py
│   │   ├── multihead_attention.py
│   │   ├── positional_encoding.py
│   │   ├── qk_norm.py
│   │   ├── residual_connection.py
│   │   ├── rms_norm.py
│   │   └── swiglu.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── pretrain_decoder_dataset.py
│   ├── pretraining/
│   │   ├── pretrain_decoder_model.py
│   │   └── pretrain_decoder.py
│   ├── training/
│   │   ├── train.py
│   └── transformer/
│       ├── decoder_layer.py
│       ├── decoder.py
│       ├── encoder_layer.py
│       ├── encoder.py
│       └── transformer.py
├── outputs/
│   └── train_wmt.log
├── requirements.txt
└── tokenizer/
    ├── data/
    │   ├── en.rar
    │   └── ru.rar
    ├── notebooks/
    │   ├── en-tokenizer-notebook.ipynb
    │   └── ru-tokenizer-notebook.ipynb
    ├── tokenizer.py
    └── vocabs/
        ├── en-vocab/
        │   ├── en_id_to_token.json
        │   └── en_token_to_id.json
        └── ru-vocab/
            ├── ru_id_to_token.json
            └── ru_token_to_id.json
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yageraaa/russian-english-transformer.git
cd russian-english-transformer
```

### 2. Set Up Docker Permissions

Ensure your user has Docker permissions to avoid `permission denied` errors (required for running the Streamlit application):
```bash
sudo usermod -aG docker $USER
newgrp docker
```
Log out and log back in to apply changes. If issues persist, use `sudo` for Docker commands.

### 3. Create `.env` File

Create a `.env` file in the `api/` directory (required for running the Streamlit application):
```bash
mkdir -p api
nano api/.env
```

Add the following content:
```
AWS_BUCKET=transformer
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123
AWS_S3_ENDPOINT_URL=http://minio:9000
AWS_ENDPOINT_URL=http://minio:9000
DATABASE_URL=postgresql://<db_user>:<db_password>@postgres:5432/translator_db
SECRET_KEY=<your_secret_key>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

**Explanation of `.env` Variables**:
- `AWS_BUCKET`: Name of the MinIO bucket. Default: `transformer`.
- `AWS_REGION`: AWS region for MinIO compatibility. Use `us-east-1`.
- `AWS_ACCESS_KEY_ID`: MinIO access key. Default: `minioadmin`.
- `AWS_SECRET_ACCESS_KEY`: MinIO secret key. Default: `minioadmin123`.
- `AWS_S3_ENDPOINT_URL` and `AWS_ENDPOINT_URL`: MinIO endpoint within Docker network. Use `http://minio:9000`.
- `DATABASE_URL`: PostgreSQL connection string. Format: `postgresql://<username>:<password>@postgres:5432/translator_db`. Choose your own `<username>` and `<password>`. Example: `postgresql://user:password123@postgres:5432/translator_db`.
- `SECRET_KEY`: JWT secret key for authentication. Generate a secure key:
  ```bash
  openssl rand -hex 32
  ```
  Example output: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6`. Do not reuse this example.
- `ALGORITHM`: JWT algorithm. Use `HS256`.
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time in minutes. Default: `30`.

**Note**: The `frontend` service uses `API_URL=http://backend:8000` defined in `docker-compose.yml`, which should resolve correctly within the Docker network `app-network`.

### 4. Obtain Model Weights

Create the `checkpoints` directory and manually place the pre-trained model weights (`transformer_00.pt`) in:
```
checkpoints/transformer_00.pt
```

```bash
mkdir -p checkpoints
```

Manually copy the weights file into the `checkpoints/` directory. Alternatively, train the model as described in the "Train the Model" section below.

### 5. Update `config.yaml`

Edit `models/configs/config.yaml` to point to the model weights:
```bash
nano models/configs/config.yaml
```

Ensure the `model_weights` path is:
```yaml
model_weights: ./checkpoints/transformer_00.pt
```

For **training**, set `base_dir` to the project root:
```yaml
base_dir: ./
```

For **running the application**, if `base_dir` is set to `/app/api` in the Docker context, verify compatibility with `api/ml/translator.py`. If needed, keep:
```yaml
base_dir: /app/api
```

## Usage

### Option 1: Run the Streamlit Application

**Note**: You do not need to create a Conda environment or install Python dependencies to run the Streamlit application, as it is fully containerized with Docker.

1. **Build and Start Containers**
   ```bash
   docker compose up --build -d
   ```
   If `permission denied` occurs, use:
   ```bash
   sudo docker compose up --build -d
   ```

2. **Verify Containers**
   ```bash
   docker ps
   ```
   Expected output:
   ```
   CONTAINER ID   IMAGE                                  STATUS                    PORTS                                                             NAMES
   <id>           russian-english-transformer-backend     Up (healthy)              0.0.0.0:8000->8000/tcp                                            russian-english-transformer-backend-1
   <id>           russian-english-transformer-frontend    Up                        0.0.0.0:8501->8501/tcp                                            russian-english-transformer-frontend-1
   <id>           postgres:16                            Up (healthy)              0.0.0.0:5432->5432/tcp                                            russian-english-transformer-postgres-1
   <id>           quay.io/minio/minio:latest             Up (healthy)              0.0.0.0:9000-9001->9000-9001/tcp                                  russian-english-transformer-minio-1
   ```

3. **Check Logs**
   ```bash
   docker compose logs frontend
   ```
   Look for:
   ```
   You can now view your Streamlit app in your browser.
   URL: http://0.0.0.0:8501
   ```

4. **Access the Application**
   - Open `http://localhost:8501` in your browser.
   - **Login**: Use a test user (e.g., `testuser`, `testpassword`). Register a new user if needed via the "Register" page.
   - **Translate**: Enter Russian text (e.g., `еда`) to get English translation (`food`).
   - **File Upload**: Upload a `.txt` file (e.g., `echo "Привет, мир!" > test.txt`) to translate and store in MinIO at `s3://transformer/translations/<username>/`.

5. **Access MinIO**
   - Open `http://localhost:9001`.
   - Login: `minioadmin`, `minioadmin123`.
   - Check the `transformer` bucket for translated files.
   - Alternatively:
     ```bash
     mc alias set minio http://localhost:9000 minioadmin minioadmin123
     mc ls minio/transformer/translations/<username>/
     ```

6. **Verify PostgreSQL Logs**
   ```bash
   docker exec -it russian-english-transformer-postgres-1 psql -U <db_user> -d translator_db
   ```
   Use the `<db_user>` and `<db_password>` from `DATABASE_URL` in `.env`. Example:
   ```sql
   SELECT * FROM translation_logs WHERE input_text = 'еда';
   \q
   ```

7. **Stop Containers**
   ```bash
   docker compose down
   ```

### Option 2: Train the Model

**Note**: Training the model requires creating a Conda environment with Python 3.12.7 and installing dependencies from `requirements.txt` in the project root.

1. **Set Up Conda Environment**
   Ensure Miniconda or Anaconda is installed, then create a Conda environment with Python 3.12.7:
   ```bash
   conda create -n translator python=3.12.7
   conda activate translator
   ```

2. **Install Python Dependencies**
   Install dependencies from the `requirements.txt` in the project root:
   ```bash
   pip install -r requirements.txt
   ```

3. **Update `config.yaml`**
   Ensure `base_dir` is set to the project root:
   ```bash
   nano models/configs/config.yaml
   ```
   ```yaml
   base_dir: ./
   model_weights: ./checkpoints/transformer_00.pt
   ```

4. **Download or Prepare Training Data**
   Place the dataset in `tokenizer/data/` (e.g., extract `en.rar` and `ru.rar`):
   ```bash
   mkdir -p tokenizer/data
   wget <dataset_url> -O tokenizer/data/dataset.tar.gz
   tar -xvzf tokenizer/data/dataset.tar.gz -C tokenizer/data/
   ```
   Alternatively, use `tokenizer/notebooks/en-tokenizer-notebook.ipynb` and `ru-tokenizer-notebook.ipynb` to preprocess data.

5. **Run Training Script**
   ```bash
   python models/training/train.py
   ```
   This trains the model using `models/configs/config.yaml` and saves weights to `checkpoints/transformer_00.pt`.

6. **Verify Trained Model**
   Check the output weights:
   ```bash
   ls -l checkpoints/transformer_00.pt
   ```

## Troubleshooting

1. **Permission Denied Error**
   If `docker compose` fails with `permission denied`:
   ```bash
   sudo docker compose down
   sudo docker stop <container_id>
   sudo docker rm <container_id>
   ```
   Or fix Docker socket permissions:
   ```bash
   sudo chown root:docker /var/run/docker.sock
   sudo chmod 660 /var/run/docker.sock
   sudo systemctl restart docker
   ```

2. **Connection Refused Error**
   If `frontend` logs show `Connection refused`:
   ```bash
   docker compose logs frontend
   ```
   Verify network connectivity:
   ```bash
   docker exec -it russian-english-transformer-frontend-1 ping backend
   ```
   Ensure `API_URL=http://backend:8000` is correctly set in the `frontend` service in `docker-compose.yml`. If the issue persists, check if `api/frontend/app.py` is using `os.getenv("API_URL")` correctly.

3. **Streamlit Not Starting**
   Check logs:
   ```bash
   docker compose logs frontend
   ```
   Ensure `api/frontend/app.py` is accessible and `api/frontend/requirements.txt` includes `streamlit`.

4. **Model Weights Missing**
   If `transformer_00.pt` is missing, ensure it is placed in the `checkpoints/` directory or train the model as described above.

5. **Network Issues**
   Verify containers are in the same network:
   ```bash
   docker inspect russian-english-transformer-frontend-1 | grep Network
   docker inspect russian-english-transformer-backend-1 | grep Network
   ```

6. **Conda Environment Issues**
   If `pip install -r requirements.txt` fails, ensure the Conda environment is active:
   ```bash
   conda activate translator
   ```
   Verify Python version:
   ```bash
   python --version
   ```
   Expected output: `Python 3.12.7`.

## Additional Notes

- **Model Weights**: If `transformer_00.pt` is not provided, train the model or manually place it in the `checkpoints/` directory.
- **SECRET_KEY**: Generate a new `SECRET_KEY` for security using `openssl rand -hex 32`. Do not reuse example keys.
- **Training Data**: Ensure the dataset matches the format expected by `models/training/train.py` (see `models/configs/config.yaml`).
