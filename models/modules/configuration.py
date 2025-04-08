from pathlib import Path


def get_config():
    BASE_DIR = Path("/home/gera/PycharmProjects/russian-english-transformer")
    DATA_DIR = BASE_DIR / "models/data"
    TOKENIZER_DIR = BASE_DIR / "tokenizer/vocabs"

    return {
        "batch_size": 16,
        "num_epochs": 20,
        "lr": 1e-4,
        "dropout": 0.1,
        "seq_len": 350,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 4,
        "d_ff": 1024,
        "src_lang": "ru",
        "tgt_lang": "en",

        "dataset_path": {
            "train_ru": str(DATA_DIR / "ru.txt"),
            "train_en": str(DATA_DIR / "en.txt"),
        },

        "vocab_paths": {
            "ru_token_to_id": str(TOKENIZER_DIR / "ru-vocab/ru_token_to_id.json"),
            "en_token_to_id": str(TOKENIZER_DIR / "en-vocab/en_token_to_id.json"),
            "ru_id_to_token": str(TOKENIZER_DIR / "ru-vocab/ru_id_to_token.json"),
            "en_id_to_token": str(TOKENIZER_DIR / "en-vocab/en_id_to_token.json"),
        },

        "model_dir": str(BASE_DIR / "checkpoints"),
        "model_basename": "transformer_",
        "preload": "latest",
        "experiment_name": "ru-en-transformer",
        "log_interval": 50,
        "example_interval": 1,
        "log_examples": True,
        "log_dir": str(BASE_DIR / "logs")
    }


def get_weights_file_path(config, epoch: int):
    model_dir = Path(config["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir / f"{config['model_basename']}{epoch:02d}.pt")


def latest_weights_file_path(config):
    model_dir = Path(config["model_dir"])
    if not model_dir.exists():
        return None
    checkpoints = list(model_dir.glob(f"{config['model_basename']}*.pt"))
    if not checkpoints:
        return None
    checkpoints.sort()
    return str(checkpoints[-1])