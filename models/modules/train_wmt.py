import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import hydra
from omegaconf import DictConfig
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from accelerate import Accelerator
from models.modules.dataset_wmt import BilingualTranslationDataset, load_hf_dataset
from models.modules.transformer_wmt import TransformerWithNewTechniques
from tokenizer.modules.tokenizer import Tokenizer
from typing import Optional, Dict, Any, List, Tuple
import signal
import gc


@hydra.main(config_path="../configs", config_name="config_wmt", version_base="1.2")
def train_model(cfg: DictConfig):
    accelerator = Accelerator()
    device = accelerator.device

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

    accelerator.print("Loading dataset...")
    dataset = load_hf_dataset(cfg)
    train_ds, val_ds = create_datasets(cfg, tokenizer, dataset)
    accelerator.print(f"Train dataset size: {len(train_ds.dataset)}")
    accelerator.print(f"Validation dataset size: {len(val_ds.dataset)}")
    accelerator.print("Initializing model...")
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
    )

    if Path(cfg.data.decoder_weights).exists():
        accelerator.print(f"Loading pretrained decoder weights from {cfg.data.decoder_weights}...")
        load_pretrained_decoder_weights(model, cfg.data.decoder_weights, accelerator)
    else:
        accelerator.print(
            f"Warning: Decoder weights file {cfg.data.decoder_weights} not found. Using random initialization.")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.en_token_to_id['<pad>'],
        label_smoothing=0.1
    )

    model, optimizer, train_ds, val_ds = accelerator.prepare(model, optimizer, train_ds, val_ds)
    epoch, global_step = load_checkpoint(cfg, model, optimizer, accelerator)
    accelerator.print(f"Starting from epoch {epoch}, global step {global_step}")

    def signal_handler(sig, frame):
        accelerator.print("Interrupt detected, saving checkpoint...")
        save_checkpoint(cfg, epoch, global_step, model, optimizer, accelerator)
        accelerator.print(f"Checkpoint saved at {get_weights_file_path(cfg, epoch)}")
        wandb.finish()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    epoch_progress = tqdm(range(epoch, cfg.training.num_epochs), desc="Training", position=0,
                          disable=not accelerator.is_local_main_process)

    for epoch in epoch_progress:
        epoch_progress.set_description(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        model.train()

        batch_iterator = tqdm(
            enumerate(train_ds),
            desc=f"Training Batch",
            total=len(train_ds),
            leave=False,
            disable=not accelerator.is_local_main_process
        )

        for batch_idx, batch in batch_iterator:
            inputs = {k: v for k, v in batch.items() if k != 'src_text' and k != 'tgt_text'}

            encoder_output = model.encode(inputs['encoder_input'], inputs['encoder_mask'])
            decoder_output = model.decode(encoder_output, inputs['encoder_mask'], inputs['decoder_input'],
                                          inputs['decoder_mask'])
            proj_output = model.project(decoder_output)
            loss = loss_fn(proj_output.view(-1, len(tokenizer.en_token_to_id)), inputs['label'].view(-1))
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            log_data = {
                "train/loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
                "step": global_step
            }

            batch_iterator.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

            if accelerator.is_local_main_process:
                wandb.log(log_data)

            global_step += 1

            if global_step % cfg.training.validation_interval_steps == 0:
                accelerator.print(f"\nRunning validation at step {global_step}...")
                val_loss, val_metrics = run_validation(model, val_ds, device, loss_fn, tokenizer, cfg, accelerator)

                if accelerator.is_local_main_process:
                    wandb.log({"val/loss": val_loss, "epoch": epoch, "step": global_step, **val_metrics})

                    if getattr(cfg.logging, 'log_examples', True):
                        wandb.log(log_translations(model, tokenizer, device, cfg, epoch, accelerator))

                save_checkpoint(cfg, epoch, global_step, model, optimizer, accelerator)
                model.train()
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        accelerator.print(f"\nRunning end-of-epoch validation...")
        val_loss, val_metrics = run_validation(model, val_ds, device, loss_fn, tokenizer, cfg, accelerator)

        if accelerator.is_local_main_process:
            wandb.log({"val/loss": val_loss, "epoch": epoch, "step": global_step, **val_metrics})

            if getattr(cfg.logging, 'log_examples', True):
                wandb.log(log_translations(model, tokenizer, device, cfg, epoch, accelerator))

        save_checkpoint(cfg, epoch, global_step, model, optimizer, accelerator)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    wandb.finish()


def load_pretrained_decoder_weights(model, weights_path, accelerator):
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        decoder_state = {}

        for key, value in state_dict.items():
            if key.startswith("decoder."):
                new_key = key[len("decoder."):]
                decoder_state[new_key] = value

        if not decoder_state:
            for key, value in state_dict.items():
                if key.startswith("layers.") or key.startswith("embed.") or key.startswith("pos_encoding"):
                    decoder_state[key] = value

        if not decoder_state:
            accelerator.print(f"[!] No decoder weights found in {weights_path}")
            return

        model.decoder.load_state_dict(decoder_state, strict=False)
        accelerator.print(f"[✓] Decoder weights loaded ({len(decoder_state)} keys)")

    except Exception as e:
        accelerator.print(f"[X] Failed to load decoder weights: {e}")


def create_datasets(cfg: DictConfig, tokenizer, dataset):
    train_dataset = BilingualTranslationDataset(
        dataset, tokenizer, cfg.language.src_lang, cfg.language.tgt_lang, cfg.training.seq_len,
        split=cfg.dataset.train_split)
    val_dataset = BilingualTranslationDataset(
        dataset, tokenizer, cfg.language.src_lang, cfg.language.tgt_lang, cfg.training.seq_len,
        split=cfg.dataset.validation_split)

    return (
        DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, pin_memory=True),
        DataLoader(val_dataset, batch_size=cfg.training.batch_size, pin_memory=True)
    )


def load_checkpoint(cfg: DictConfig, model, optimizer, accelerator):
    if cfg.logging.preload == "latest":
        if model_file := latest_weights_file_path(cfg):
            try:
                accelerator.print(f"Loading checkpoint from {model_file}...")
                checkpoint = torch.load(model_file, map_location='cpu')
                accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                if "accelerator_state" in checkpoint:
                    accelerator.load_state_dict(checkpoint["accelerator_state"])

                accelerator.print(f"Checkpoint loaded successfully")
                return checkpoint["epoch"] + 1, checkpoint["global_step"]
            except Exception as e:
                accelerator.print(f"Error loading checkpoint: {e}")
                accelerator.print("Starting from scratch")
                return 0, 0
    return 0, 0


def save_checkpoint(cfg: DictConfig, epoch, step, model, optimizer, accelerator):
    model_file = get_weights_file_path(cfg, epoch)

    if accelerator.is_local_main_process:
        accelerator.print(f"Saving checkpoint to {model_file}...")

        accelerator.save({
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accelerator_state": accelerator.get_state_dict()
        }, model_file)

        accelerator.print(f"Checkpoint saved successfully")


def decode_until_end(ids, id_to_token, end_token="<end>"):
    tokens = []
    for idx in ids:
        token = id_to_token.get(idx, '<unk>')
        if token == end_token:
            break
        tokens.append(token)
    return " ".join(tokens)


def run_validation(model, val_loader, device, loss_fn, tokenizer, cfg, accelerator):
    model.eval()
    total_loss = 0
    total_bleu = 0
    total_samples = 0
    max_samples = cfg.training.validation_samples

    val_iterator = tqdm(
        enumerate(val_loader),
        desc="Validation",
        total=min(len(val_loader), max_samples // cfg.training.batch_size + 1),
        leave=False,
        disable=not accelerator.is_local_main_process
    )

    with torch.no_grad():
        for batch_idx, batch in val_iterator:
            if total_samples >= max_samples:
                break

            inputs = {k: v for k, v in batch.items() if k != 'src_text' and k != 'tgt_text'}

            encoder_output = model.encode(inputs['encoder_input'], inputs['encoder_mask'])
            decoder_output = model.decode(
                encoder_output,
                inputs['encoder_mask'],
                inputs['decoder_input'],
                inputs['decoder_mask']
            )
            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(-1, len(tokenizer.en_token_to_id)),
                inputs['label'].view(-1)
            ).item()

            total_loss += loss

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
                bleu_score = calculate_bleu(pred, ref)
                total_bleu += bleu_score
                total_samples += 1

            val_iterator.set_postfix(loss=f"{loss:.4f}", samples=total_samples)

    avg_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0
    avg_bleu = total_bleu / total_samples if total_samples > 0 else 0

    accelerator.print(f"Validation results: Loss = {avg_loss:.4f}, BLEU = {avg_bleu:.4f}, Samples = {total_samples}")

    metrics = {
        "val/bleu": avg_bleu,
        "val/samples": total_samples
    }

    return avg_loss, metrics


def calculate_bleu(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = [reference.split()]
    return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=SmoothingFunction().method1)


def log_translations(model, tokenizer, device, cfg: DictConfig, epoch: int, accelerator):
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

            translation = tokenizer.decode_ids(
                output[0].cpu().numpy(),
                getattr(tokenizer, f"{cfg.language.tgt_lang}_id_to_token")
            )

            accelerator.print(f"Epoch {epoch} - Source: {src}, Reference: {ref}, Translation: {translation}")
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
