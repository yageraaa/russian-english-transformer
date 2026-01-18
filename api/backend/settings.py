import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    AWS_BUCKET: str = os.getenv("AWS_BUCKET")
    AWS_REGION: str = os.getenv("AWS_REGION")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_HOST: str = os.getenv("AWS_HOST")
    AWS_ENDPOINT_URL: str = os.getenv("AWS_ENDPOINT_URL")
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

settings = Settings()