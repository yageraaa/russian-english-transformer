from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from api.backend.settings import settings
import hashlib

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    if pwd_context.verify(plain_password, hashed_password):
        return True
    
    if len(plain_password.encode('utf-8')) > 72:
        hashed_plain = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
        return pwd_context.verify(hashed_plain, hashed_password)
    
    return False

def get_password_hash(password):
    if len(password.encode('utf-8')) > 72:
        password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt