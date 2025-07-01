from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import torch
from tokenizer.tokenizer import Tokenizer
from models.transformer.transformer_wmt import TransformerWithNewTechniques
from omegaconf import OmegaConf
from pathlib import Path
from api.backend.settings import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

CONFIG_PATH = "/home/gera/PycharmProjects/russian-english-transformer/models/configs/config_wmt.yaml"
cfg = OmegaConf.load(CONFIG_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer({
    'ru_token_to_id': cfg.vocabs.ru_token_to_id,
    'ru_id_to_token': cfg.vocabs.ru_id_to_token,
    'en_token_to_id': cfg.vocabs.en_token_to_id,
    'en_id_to_token': cfg.vocabs.en_id_to_token
})

model = TransformerWithNewTechniques(
    src_vocab_size=len(tokenizer.ru_token_to_id),
    tgt_vocab_size=len(tokenizer.en_token_to_id),
    src_seq_len=cfg.training.seq_len,
    tgt_seq_len=cfg.training.seq_len,
    d_model=cfg.model.d_model,
    num_layers=cfg.model.num_layers,
    num_heads=cfg.model.num_heads,
    dropout=cfg.training.dropout,
    d_ff=cfg.model.d_ff
).to(device)

model_weights_path = cfg.data.model_weights
checkpoint = torch.load(model_weights_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
model.eval()

def translate_text(input_text: str) -> str:
    input_tokens = tokenizer.encode_text(input_text.lower(), tokenizer.ru_token_to_id, tokenizer.ru_vocab)
    encoder_input = torch.tensor([input_tokens], dtype=torch.int64).to(device)
    with torch.no_grad():
        output = model.translate_batch(
            encoder_input,
            start_token_id=tokenizer.en_token_to_id['<start>'],
            end_token_id=tokenizer.en_token_to_id['<end>'],
            max_len=cfg.training.seq_len
        )
    return tokenizer.decode_ids(output[0].cpu().numpy(), tokenizer.en_id_to_token)