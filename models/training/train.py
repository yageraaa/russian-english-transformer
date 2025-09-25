import mlflow
import mlflow.pytorch
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
from models.data.dataset import BilingualTranslationDataset, load_hf_dataset
from models.transformer.transformer import TransformerWithNewTechniques
from tokenizer.tokenizer import Tokenizer
from typing import Optional
import signal
import gc
import re
import random
import numpy as np



@hydra.main(config_path="../../models/configs", config_name="config", version_base="1.2")
def train_model(cfg: DictConfig):
    accelerator = Accelerator(
        mixed_precision=getattr(cfg.training, 'mixed_precision', 'no'),
        gradient_accumulation_steps=getattr(cfg.training, 'gradient_accumulation_steps', 1),
        log_with="mlflow"
    )
    device = accelerator.device

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("transformer-ru-en")
    mlflow.start_run(run_name=f"transformer-ru-en-{int(time())}")
    mlflow.log_params(hydra.utils.instantiate(cfg))

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
    accelerator.print(f"Using device: {device}")
    accelerator.print(f"Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            accelerator.print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

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

    if cfg.training.use_pretrained_decoder and Path(cfg.data.decoder_weights).exists():
        accelerator.print(f"Loading pretrained decoder weights from {cfg.data.decoder_weights}...")
        load_pretrained_decoder_weights(model, cfg.data.decoder_weights, accelerator)
    else:
        accelerator.print(
            f"Warning: Using random initialization for decoder. "
            f"{'Pretrained decoder weights disabled' if not cfg.training.use_pretrained_decoder else f'Decoder weights file {cfg.data.decoder_weights} not found.'}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.en_token_to_id['<pad>'],
        label_smoothing=0.1
    )

    scaler = torch.amp.GradScaler('cuda', enabled=False)
    if cfg.training.mixed_precision == 'bf16':
        accelerator.print("Using bf16 mixed precision training")
    elif cfg.training.mixed_precision == 'fp16':
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        accelerator.print("Using fp16 mixed precision training")

    model, optimizer, train_ds, val_ds = accelerator.prepare(model, optimizer, train_ds, val_ds)

    def safe_encode(*args, **kwargs):
        return model.module.encode(*args, **kwargs) if hasattr(model, 'module') else model.encode(*args, **kwargs)

    def safe_decode(*args, **kwargs):
        return model.module.decode(*args, **kwargs) if hasattr(model, 'module') else model.decode(*args, **kwargs)

    def safe_project(*args, **kwargs):
        return model.module.project(*args, **kwargs) if hasattr(model, 'module') else model.project(*args, **kwargs)

    def safe_translate_batch(*args, **kwargs):
        return model.module.translate_batch(*args, **kwargs) if hasattr(model, 'module') else model.translate_batch(
            *args, **kwargs)

    model.encode = safe_encode
    model.decode = safe_decode
    model.project = safe_project
    model.translate_batch = safe_translate_batch

    epoch, global_step = load_checkpoint(cfg, model, optimizer, accelerator)
    accelerator.print(f"Starting from epoch {epoch}, global step {global_step}")

    def signal_handler(sig, frame):
        accelerator.print("Interrupt detected, saving checkpoint...")
        save_checkpoint(cfg, epoch, global_step, model, optimizer, accelerator)
        accelerator.print(f"Checkpoint saved at {get_weights_file_path(cfg, epoch)}")

        accelerator.print("Cleaning up GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                accelerator.print(f"GPU {i} memory usage: {memory_allocated:.2f} GB")

        mlflow.end_run()
        accelerator.print("Training stopped safely.")
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
            desc="Training Batch",
            total=len(train_ds),
            leave=False,
            disable=not accelerator.is_local_main_process
        )

        for batch_idx, batch in batch_iterator:
            try:
                inputs = {k: v for k, v in batch.items() if k != 'src_text' and k != 'tgt_text'}

                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if cfg.training.mixed_precision == 'bf16' else torch.float16 if cfg.training.mixed_precision == 'fp16' else torch.float32):
                    encoder_output = model.encode(inputs['encoder_input'], inputs['encoder_mask'])
                    decoder_output = model.decode(encoder_output, inputs['encoder_mask'], inputs['decoder_input'],
                                                  inputs['decoder_mask'])
                    proj_output = model.project(decoder_output)
                    loss = loss_fn(proj_output.view(-1, len(tokenizer.en_token_to_id)), inputs['label'].view(-1))

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
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
                    mlflow.log_metrics(log_data, step=global_step)

                global_step += 1

                if global_step % cfg.training.validation_interval_steps == 0:
                    accelerator.print(f"\nRunning validation at step {global_step}...")
                    try:
                        accelerator.wait_for_everyone()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        val_loss, val_metrics = run_validation(model, val_ds, device, loss_fn, tokenizer, cfg, accelerator)

                        if accelerator.is_local_main_process:
                            mlflow.log_metrics({"val/loss": val_loss, "epoch": epoch, "step": global_step, **val_metrics},
                                               step=global_step)

                            if getattr(cfg.logging, 'log_examples', True):
                                log_translations_mlflow(model, tokenizer, device, cfg, epoch, accelerator)

                        save_checkpoint(cfg, epoch, global_step, model, optimizer, accelerator)
                        model.train()
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                    except Exception as e:
                        accelerator.print(f"Error during validation at step {global_step}: {e}")
                        accelerator.print("Skipping validation and continuing training...")
                        try:
                            save_checkpoint(cfg, epoch, global_step, model, optimizer, accelerator)
                        except Exception as checkpoint_error:
                            accelerator.print(f"Error saving checkpoint: {checkpoint_error}")
                        
                        model.train()
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                accelerator.print(f"Error in training batch {batch_idx} at step {global_step}: {e}")
                accelerator.print("Skipping this batch and continuing...")
                optimizer.zero_grad()
                global_step += 1
                continue

        accelerator.print(f"\nRunning end-of-epoch validation...")
        try:
            accelerator.wait_for_everyone()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            val_loss, val_metrics = run_validation(model, val_ds, device, loss_fn, tokenizer, cfg, accelerator)

            if accelerator.is_local_main_process:
                mlflow.log_metrics({"val/loss": val_loss, "epoch": epoch, "step": global_step, **val_metrics},
                                   step=global_step)

                if getattr(cfg.logging, 'log_examples', True):
                    log_translations_mlflow(model, tokenizer, device, cfg, epoch, accelerator)

            save_checkpoint(cfg, epoch, global_step, model, optimizer, accelerator)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            accelerator.print(f"Error during end-of-epoch validation: {e}")
            accelerator.print("Skipping validation but saving checkpoint...")
            try:
                save_checkpoint(cfg, epoch, global_step, model, optimizer, accelerator)
            except Exception as checkpoint_error:
                accelerator.print(f"Error saving checkpoint: {checkpoint_error}")
            
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    mlflow.end_run()

    accelerator.print("Training completed. Cleaning up GPU memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            accelerator.print(f"GPU {i} final memory usage: {memory_allocated:.2f} GB")

    accelerator.print("Training finished successfully!")


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
        accelerator.print(f"[âœ“] Decoder weights loaded ({len(decoder_state)} keys)")

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
        DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=getattr(cfg.training, 'num_workers', 0)
        ),
        DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            pin_memory=True,
            num_workers=getattr(cfg.training, 'num_workers', 0)
        )
    )


def load_checkpoint(cfg: DictConfig, model, optimizer, accelerator):
    if cfg.logging.preload == "latest":
        if model_file := latest_weights_file_path(cfg):
            try:
                accelerator.print(f"Loading checkpoint from {model_file}...")
                checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                if "accelerator_state" in checkpoint:
                    try:
                        accelerator.load_state(checkpoint["accelerator_state"])
                    except AttributeError:
                        accelerator.print("Warning: Accelerator state loading not supported, continuing without it")
                    except Exception as e:
                        accelerator.print(f"Warning: Failed to load accelerator state: {e}")

                accelerator.print(f"Checkpoint loaded successfully")
                return checkpoint["epoch"] + 1, checkpoint["global_step"]
            except Exception as e:
                accelerator.print(f"Error loading checkpoint: {e}")
                accelerator.print("Starting from scratch")
                return 0, 0
    return 0, 0


def save_checkpoint(cfg: DictConfig, epoch, step, model, optimizer, accelerator):
    try:
        model_dir = Path(cfg.data.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg
        }
        
        try:
            checkpoint_data["accelerator_state"] = accelerator.get_state_dict(model)
        except Exception as e:
            accelerator.print(f"Warning: Could not save accelerator state: {e}")
        
        if accelerator.is_local_main_process:
            accelerator.print(f"Saving checkpoint to {model_dir}...")
            
            checkpoint_path = model_dir / f"{cfg.logging.model_basename}epoch_{epoch:02d}_step_{step:06d}.pt"
            accelerator.save(checkpoint_data, str(checkpoint_path))
            
            latest_path = model_dir / f"{cfg.logging.model_basename}latest.pt"
            accelerator.save(checkpoint_data, str(latest_path))
            
            accelerator.print(f"Checkpoint saved successfully: {checkpoint_path}")
            accelerator.print(f"Latest checkpoint: {latest_path}")
            
            cleanup_old_checkpoints(cfg, keep_last_n=5)
            
    except Exception as e:
        accelerator.print(f"Error saving checkpoint: {e}")
        accelerator.print("Checkpoint save failed, but training will continue...")


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
            try:
                if total_samples >= max_samples:
                    break

                inputs = {k: v for k, v in batch.items() if k != 'src_text' and k != 'tgt_text'}

                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if cfg.training.mixed_precision == 'bf16' else torch.float16 if cfg.training.mixed_precision == 'fp16' else torch.float32):
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
                    max_len=cfg.training.seq_len,
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
                
            except Exception as e:
                accelerator.print(f"Error in validation batch {batch_idx}: {e}")
                accelerator.print("Skipping this validation batch...")
                continue

    avg_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0
    avg_bleu = total_bleu / total_samples if total_samples > 0 else 0

    accelerator.print(f"Validation results: Loss = {avg_loss:.4f}, BLEU = {avg_bleu:.4f}, Samples = {total_samples}")

    metrics = {
        "val/bleu": avg_bleu,
        "val/samples": total_samples
    }

    return avg_loss, metrics


def calculate_bleu(prediction: str, reference: str) -> float:
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    pred_normalized = normalize_text(prediction)
    ref_normalized = normalize_text(reference)

    pred_tokens = pred_normalized.split()
    ref_tokens = ref_normalized.split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)





def cleanup_old_checkpoints(cfg: DictConfig, keep_last_n: int = 5):
    model_dir = Path(cfg.data.model_dir)
    if not model_dir.exists():
        return
    
    checkpoint_files = list(model_dir.glob(f"{cfg.logging.model_basename}epoch_*_step_*.pt"))
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for old_checkpoint in checkpoint_files[keep_last_n:]:
        try:
            old_checkpoint.unlink()
        except Exception as e:
            print(f"Failed to delete old checkpoint {old_checkpoint}: {e}")


def log_translations_mlflow(model, tokenizer, device, cfg: DictConfig, epoch: int, accelerator, step: int = None):
    try:
        examples = [
            ("Привет, как дела?", "Hello, how are you?"),
            ("Сегодня хорошая погода.", "The weather is nice today."),
            ("Собака гуляет в парке.", "Dog is walking in the park."),
            ("Я люблю читать книги.", "I love reading books."),
            ("Машина стоит на улице.", "The car is parked on the street."),
            ("Ребенок играет во дворе.", "The child is playing in the yard."),
            ("Мы идем в магазин.", "We are going to the store."),
            ("Кошка спит на диване.", "The cat is sleeping on the sofa."),
            ("Время обедать.", "It's time for lunch."),
            ("Дом большой и красивый.", "The house is big and beautiful."),
            ("Он работает в офисе.", "He works in the office."),
            ("Она готовит ужин.", "She is cooking dinner."),
            ("Дети учатся в школе.", "Children study at school."),
            ("Птицы поют в саду.", "Birds are singing in the garden."),
            ("Книга лежит на столе.", "The book is on the table.")
        ]

        model.eval()
        translations = []

        with torch.no_grad():
            for src, ref in examples:
                try:
                    input_tokens = tokenizer.encode_text(
                        src,
                        getattr(tokenizer, f"{cfg.language.src_lang}_token_to_id"),
                        getattr(tokenizer, f"{cfg.language.src_lang}_vocab")
                    )
                    encoder_input = torch.tensor([input_tokens], dtype=torch.int64).to(device)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16 if cfg.training.mixed_precision == 'bf16' else torch.float16 if cfg.training.mixed_precision == 'fp16' else torch.float32):
                        output = model.translate_batch(
                            encoder_input,
                            max_len=cfg.training.seq_len,
                            start_token_id=tokenizer.en_token_to_id['<start>'],
                            end_token_id=tokenizer.en_token_to_id['<end>']
                        )

                    translation = tokenizer.decode_ids(
                        output[0].cpu().numpy(),
                        getattr(tokenizer, f"{cfg.language.tgt_lang}_id_to_token")
                    )

                    accelerator.print(f"Epoch {epoch} Step {step} - Source: {src}, Reference: {ref}, Translation: {translation}")
                    translations.append([src, ref, translation])
                    
                except Exception as e:
                    accelerator.print(f"Error translating example '{src}': {e}")
                    translations.append([src, ref, "Translation failed"])

        step_info = f"_step_{step:06d}" if step is not None else ""
        translation_text = f"Epoch {epoch}{step_info} Translations:\n"
        for i, (src, ref, trans) in enumerate(translations):
            bleu_score = calculate_bleu(trans, ref)
            translation_text += f"Example {i+1}:\nSource: {src}\nReference: {ref}\nTranslation: {trans}\nBLEU: {bleu_score:.4f}\n\n"

        mlflow.log_text(translation_text, f"translations_epoch_{epoch}{step_info}.txt")

        for i, (src, ref, trans) in enumerate(translations):
            bleu_score = calculate_bleu(trans, ref)
            mlflow.log_metric(f"example_{i+1}_bleu", bleu_score, step=step if step else epoch)
            mlflow.log_metric(f"example_{i+1}_bleu_epoch", bleu_score, step=epoch)
            
    except Exception as e:
        accelerator.print(f"Error in log_translations_mlflow: {e}")
        accelerator.print("Skipping translation logging...")


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