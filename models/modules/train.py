import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from time import time
import hydra
from omegaconf import DictConfig
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from models.modules.dataset import BilingualTranslationDataset, load_local_dataset
from models.modules.transformer import Transformer
from tokenizer.modules.tokenizer import Tokenizer
from typing import Optional

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def train_model(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="transformer-ru-en",
        name=f"{cfg.logging.experiment_name}-{int(time())}",
        config=hydra.utils.instantiate(cfg)
    )

    tokenizer = Tokenizer({
        'ru_token_to_id': cfg.vocabs.ru_token_to_id,
        'ru_id_to_token': cfg.vocabs.ru_id_to_token,
        'en_token_to_id': cfg.vocabs.en_token_to_id,
        'en_id_to_token': cfg.vocabs.en_id_to_token
    })

    train_ds, val_ds = create_datasets(cfg, tokenizer)

    model = Transformer(
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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.en_token_to_id['<pad>'],
        label_smoothing=0.1
    )

    epoch, global_step = load_checkpoint(cfg, model, optimizer)

    epoch_progress = tqdm(range(epoch, cfg.training.num_epochs), desc="Training", position=0)

    for epoch in epoch_progress:
        epoch_progress.set_description(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        model.train()

        for batch in tqdm(train_ds, desc="..", leave=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'src_text' and k != 'tgt_text'}
            encoder_output = model.encode(inputs['encoder_input'], inputs['encoder_mask'])
            decoder_output = model.decode(encoder_output, inputs['encoder_mask'], inputs['decoder_input'],
                                          inputs['decoder_mask'])
            proj_output = model.project(decoder_output)
            loss = loss_fn(proj_output.view(-1, len(tokenizer.en_token_to_id)), inputs['label'].view(-1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            log_data = {
                "train/loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
                "step": global_step
            }

            wandb.log(log_data)
            global_step += 1

        val_loss, val_metrics = run_validation(model, val_ds, device, loss_fn, tokenizer, cfg)
        wandb.log({"val/loss": val_loss, "epoch": epoch, **val_metrics})
        if getattr(cfg.logging, 'log_examples', True):
            wandb.log(log_translations(model, tokenizer, device, cfg, epoch))
        save_checkpoint(cfg, epoch, global_step, model, optimizer)

    wandb.finish()


def create_datasets(cfg: DictConfig, tokenizer):
    train_data = load_local_dataset(cfg.dataset.train_ru, cfg.dataset.train_en)
    # train_data = load_local_dataset(cfg.dataset.train_ru, cfg.dataset.train_en)[:10000]
    # train, val = random_split(train_data, [int(0.9 * len(train_data)), len(train_data) - int(0.9 * len(train_data))])
    val_size = 10000
    train_size = len(train_data) - val_size
    train, val = random_split(train_data, [train_size, val_size])

    return (
        DataLoader(
            BilingualTranslationDataset(
                train, tokenizer, cfg.language.src_lang, cfg.language.tgt_lang, cfg.training.seq_len),
            batch_size=cfg.training.batch_size, shuffle=True, pin_memory=True),
        DataLoader(
            BilingualTranslationDataset(
                val, tokenizer, cfg.language.src_lang, cfg.language.tgt_lang, cfg.training.seq_len),
            batch_size=cfg.training.batch_size, pin_memory=True)
    )


def load_checkpoint(cfg: DictConfig, model, optimizer):
    if cfg.logging.preload == "latest":
        if model_file := latest_weights_file_path(cfg):
            state = torch.load(model_file)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            return state["epoch"] + 1, state["global_step"]
    return 0, 0


def save_checkpoint(cfg: DictConfig, epoch, step, model, optimizer):
    model_file = get_weights_file_path(cfg, epoch)
    torch.save({
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, model_file)

def decode_until_end(ids, id_to_token, end_token="<end>"):
    tokens = []
    for idx in ids:
        token = id_to_token.get(idx, '<unk>')
        if token == end_token:
            break
        tokens.append(token)
    return " ".join(tokens)

def run_validation(model, val_loader, device, loss_fn, tokenizer, cfg):
    model.eval()
    total_loss = 0
    total_bleu = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'src_text' and k != 'tgt_text'}
            encoder_output = model.encode(inputs['encoder_input'], inputs['encoder_mask'])
            decoder_output = model.decode(
                encoder_output,
                inputs['encoder_mask'],
                inputs['decoder_input'],
                inputs['decoder_mask']
            )
            proj_output = model.project(decoder_output)

            total_loss += loss_fn(
                proj_output.view(-1, len(tokenizer.en_token_to_id)),
                inputs['label'].view(-1)
            ).item()

            translated = model.translate_batch(
                inputs['encoder_input'],
                start_token_id=tokenizer.en_token_to_id['<start>'],
                end_token_id=tokenizer.en_token_to_id['<end>']
            )

            for i in range(translated.size(0)):
                pred = decode_until_end(
                    translated[i].cpu().numpy(),
                    getattr(tokenizer, f"{cfg.language.tgt_lang}_id_to_token"),
                    end_token="<end>"
                )
                ref = batch['tgt_text'][i]
                total_bleu += calculate_bleu(pred, ref)
                total_samples += 1

    metrics = {
        "val/bleu": total_bleu / total_samples
    }
    return total_loss / len(val_loader), metrics


def calculate_bleu(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = [reference.split()]
    return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=SmoothingFunction().method1)


def log_translations(model, tokenizer, device, cfg: DictConfig, epoch: int):
    examples = [
        ("Привет, как дела?", "Hello, how are you?"),
        ("Сегодня хорошая погода.", "The weather is nice today."),
        ("Собака гуляет в парке.", "Dog is walking in the park.")
    ]

    model.eval()
    translations = []

    with torch.no_grad():
        for src, ref in examples:
            input_tokens = tokenizer.encode_text(
                src,
                getattr(tokenizer, f"{cfg.language.src_lang}_token_to_id"),
                getattr(tokenizer, f"{cfg.language.src_lang}_vocab")
            )
            encoder_input = torch.tensor([input_tokens], dtype=torch.int64).to(device)
            output = model.translate_batch(
                encoder_input,
                start_token_id=tokenizer.en_token_to_id['<start>'],
                end_token_id=tokenizer.en_token_to_id['<end>']
            )
            translation = tokenizer.decode_ids(output[0].cpu().numpy(), getattr(tokenizer, f"{cfg.language.tgt_lang}_id_to_token"))
            print(f"Epoch {epoch} - Source: {src}, Reference: {ref}, Translation: {translation}")
            translations.append([src, ref, translation])

    return {f"examples_epoch_{epoch}": wandb.Table(columns=["Source", "Reference", "Translation"], data=translations)}


def get_weights_file_path(cfg: DictConfig, epoch: int) -> str:
    model_dir = Path(cfg.data.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir / f"{cfg.logging.model_basename}{epoch:02d}.pt")


def latest_weights_file_path(cfg: DictConfig) -> Optional[str]:
    model_dir = Path(cfg.data.model_dir)
    if not model_dir.exists():
        return None
    checkpoints = list(model_dir.glob(f"{cfg.logging.model_basename}*.pt"))
    if not checkpoints:
        return None
    checkpoints.sort()
    return str(checkpoints[-1])


if __name__ == "__main__":
    train_model()