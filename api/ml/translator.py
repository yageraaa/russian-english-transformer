import torch
from pathlib import Path
from omegaconf import OmegaConf
from tokenizer.tokenizer import Tokenizer
from models.transformer.transformer import TransformerWithNewTechniques
from api.backend.settings import settings

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "models" / "configs" / "config.yaml"
cfg = OmegaConf.load(CONFIG_PATH)

model_weights_path = Path(cfg.data.model_weights)
if not model_weights_path.is_absolute():
    model_weights_path = BASE_DIR / model_weights_path
model_weights_path = model_weights_path.resolve()

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

checkpoint = torch.load(model_weights_path, map_location=device, weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
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