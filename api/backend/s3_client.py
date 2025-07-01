import boto3
import botocore
import logging
from api.backend.settings import settings

class S3:
    def __init__(self):
        self.session = boto3.session.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.s3client = self.session.client(
            service_name='s3',
            endpoint_url=settings.AWS_ENDPOINT_URL,
            config=botocore.client.Config(signature_version='s3v4')
        )
        self.create_bucket(settings.AWS_BUCKET)

    def has_file(self, fileid: str):
        try:
            self.s3client.head_object(Bucket=settings.AWS_BUCKET, Key=fileid)
            return True
        except botocore.exceptions.ClientError:
            return False

    def upload_file(self, file, fileid: str):
        try:
            self.s3client.upload_fileobj(file, settings.AWS_BUCKET, fileid)
            logging.info(f"Файл загружен в S3: {fileid}")
        except botocore.exceptions.ClientError as e:
            logging.error(f"Ошибка загрузки файла в S3: {e}")
            raise

    def download_file(self, file, fileid: str):
        try:
            self.s3client.head_object(Bucket=settings.AWS_BUCKET, Key=fileid)
            self.s3client.download_fileobj(settings.AWS_BUCKET, fileid, file)
            file.seek(0)
            logging.info(f"Файл скачан из S3: {fileid}")
        except botocore.exceptions.ClientError:
            logging.error(f"Файл не найден в S3: {fileid}")
            raise FileNotFoundError("File not found")

    def delete_file(self, fileid: str):
        try:
            self.s3client.delete_object(Bucket=settings.AWS_BUCKET, Key=fileid)
            logging.info(f"Файл удалён из S3: {fileid}")
        except botocore.exceptions.ClientError as e:
            logging.error(f"Ошибка удаления файла из S3: {e}")
            raise

    def create_bucket(self, name):
        try:
            try:
                self.s3client.head_bucket(Bucket=name)
                logging.info(f"Бакет уже существует: {name}")
                return
            except botocore.exceptions.ClientError:
                pass
            self.s3client.create_bucket(Bucket=name)
            logging.info(f"Бакет создан: {name}")
        except botocore.exceptions.ClientError as e:
            logging.error(f"Ошибка создания бакета: {e}")
            raise

    def generate_link(self, bucket, key):
        try:
            url = self.s3client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=3600
            )
            logging.info(f"Сгенерирована временная ссылка для {key}")
            return url
        except botocore.exceptions.ClientError as e:
            logging.error(f"Ошибка генерации ссылки: {e}")
            raise

s3 = S3()