import logging
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from datetime import datetime
from api.backend.database import SessionLocal, User, AuthLog, TranslationLog, init_db
from api.backend.models import UserCreate, UserLogin, Token, TranslationResponse
from api.backend.utils import verify_password, get_password_hash, create_access_token
from api.ml.translator import translate_text
from api.backend.s3_client import s3
from api.backend.settings import settings
from io import BytesIO

logging.basicConfig(level=logging.INFO, filename="/tmp/uvicorn.log")
logger = logging.getLogger(__name__)

app = FastAPI(title="Russian to English Translator")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.error("Token does not contain username")
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError as e:
        logger.error(f"JWT decoding error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        logger.error(f"User {username} not found")
        raise HTTPException(status_code=401, detail="User not found")
    return user


@app.on_event("startup")
def startup_event():
    init_db()


@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        logger.error(f"Attempt to register existing user: {user.username}")
        raise HTTPException(status_code=400, detail="User already exists")
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    logger.info(f"User {user.username} registered")
    return {"message": "User registered successfully"}


@app.post("/token", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        logger.error(f"Failed login attempt for {user.username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.username})
    db_log = AuthLog(user_id=db_user.id, login_time=datetime.utcnow())
    db.add(db_log)
    db.commit()
    logger.info(f"User {user.username} logged in, token issued")
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/translate", response_model=TranslationResponse)
async def translate(
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        text = body.get("text", "").strip()
        if not text:
            logger.error("Text not provided or empty")
            raise HTTPException(status_code=400, detail="Text must not be empty")
        if len(text) > 1000:
            logger.error(f"Text exceeds 1000 characters: {len(text)}")
            raise HTTPException(status_code=400, detail="Text must not exceed 1000 characters")

        translation = translate_text(text)
        db_log = TranslationLog(user_id=current_user.id, input_text=text, request_time=datetime.utcnow())
        db.add(db_log)
        db.commit()
        return TranslationResponse(translation=translation)

    elif "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if not file:
            logger.error("File not provided")
            raise HTTPException(status_code=400, detail="File not provided")
        if file.content_type != "text/plain":
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be in .txt format")

        file_content = await file.read()
        file_text = file_content.decode("utf-8").strip()
        if not file_text:
            logger.error("File is empty")
            raise HTTPException(status_code=400, detail="File must not be empty")
        if len(file_text) > 1000:
            logger.error(f"File content exceeds 1000 characters: {len(file_text)}")
            raise HTTPException(status_code=400, detail="File content must not exceed 1000 characters")

        fileid = f"translations/{current_user.username}/{file.filename}"
        s3.upload_file(BytesIO(file_content), fileid)
        lines = file_text.splitlines()
        translations = [translate_text(line) for line in lines if line.strip()]
        translation = " ".join(translations)
        s3_path = f"s3://{settings.AWS_BUCKET}/{fileid}"

        db_log = TranslationLog(user_id=current_user.id, s3_path=s3_path, request_time=datetime.utcnow())
        db.add(db_log)
        db.commit()
        return TranslationResponse(translation=translation, s3_path=s3_path)

    else:
        logger.error(f"Unsupported Content-Type: {content_type}")
        raise HTTPException(status_code=400, detail="Only application/json and multipart/form-data are supported")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)