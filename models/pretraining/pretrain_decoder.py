import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from time import time
import hydra
from omegaconf import DictConfig
from pathlib import Path
from models.data.pretrain_decoder_dataset import EnglishLanguageModelDataset, load_english_dataset
from pretrain_decoder_model import DecoderOnlyModel
from tokenizer.tokenizer import Tokenizer
import gc


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def pretrain_decoder(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("transformer-en-decoder-pretrain")
    mlflow.start_run(run_name=f"decoder-pretrain-{int(time())}")
    mlflow.log_params(hydra.utils.instantiate(cfg))

    tokenizer = Tokenizer({
        'en_token_to_id': cfg.vocabs.en_token_to_id,
        'en_id_to_token': cfg.vocabs.en_id_to_token
    })

    # train_data = load_english_dataset(cfg.dataset.train_en)
    train_data = load_english_dataset(cfg.dataset.train_en)[:500000]
    val_size = min(10000, int(0.1 * len(train_data)))
    train_size = len(train_data) - val_size
    train, val = random_split(train_data, [train_size, val_size])

    train_ds = DataLoader(
        EnglishLanguageModelDataset(
            train, tokenizer, tgt_lang=cfg.language.tgt_lang, seq_length=cfg.training.seq_len),
        batch_size=cfg.training.batch_size, shuffle=True, pin_memory=True)

    val_ds = DataLoader(
        EnglishLanguageModelDataset(
            val, tokenizer, tgt_lang=cfg.language.tgt_lang, seq_length=cfg.training.seq_len),
        batch_size=cfg.training.batch_size, pin_memory=True)

    model = DecoderOnlyModel(
        vocab_size=len(tokenizer.en_token_to_id),
        seq_len=cfg.training.seq_len,
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

    epoch_progress = tqdm(range(epoch, cfg.training.num_epochs), desc="Pretraining decoder", position=0)

    for epoch in epoch_progress:
        epoch_progress.set_description(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        model.train()

        for batch in tqdm(train_ds, desc="Training", leave=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'tgt_text'}

            output = model(inputs['decoder_input'], inputs['decoder_mask'])

            loss = loss_fn(output.view(-1, len(tokenizer.en_token_to_id)), inputs['label'].view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            log_data = {
                "train/loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
                "step": global_step
            }

            mlflow.log_metrics(log_data, step=global_step)
            global_step += 1

        val_loss = run_validation(model, val_ds, device, loss_fn)
        mlflow.log_metrics({"val/loss": val_loss, "epoch": epoch}, step=global_step)

        if getattr(cfg.logging, 'log_examples', True) and epoch % cfg.logging.example_interval == 0:
            log_generations_mlflow(model, tokenizer, device, cfg, epoch)

        save_checkpoint(cfg, epoch, global_step, model, optimizer, prefix="decoder_pretrain_")

    mlflow.end_run()
    
    print("Pretraining completed. Cleaning up GPU memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Final memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print("Pretraining finished successfully!")


def load_checkpoint(cfg: DictConfig, model, optimizer):
    if cfg.logging.preload == "latest":
        if model_file := latest_weights_file_path(cfg, prefix="decoder_pretrain_"):
            state = torch.load(model_file)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            return state["epoch"] + 1, state["global_step"]
    return 0, 0


def save_checkpoint(cfg: DictConfig, epoch, step, model, optimizer, prefix="decoder_pretrain_"):
    model_file = get_weights_file_path(cfg, epoch, prefix)
    torch.save({
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, model_file)


def run_validation(model, val_loader, device, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'tgt_text'}
            output = model(inputs['decoder_input'], inputs['decoder_mask'])
            total_loss += loss_fn(
                output.view(-1, output.size(-1)),
                inputs['label'].view(-1)
            ).item()

    return total_loss / len(val_loader)


def log_generations_mlflow(model, tokenizer, device, cfg: DictConfig, epoch: int):
    examples = [
        "hello, how are you?",
        "the weather is nice today.",
        "i would like to learn more about"
    ]

    model.eval()
    generations = []

    with torch.no_grad():
        for text in examples:
            input_tokens = tokenizer.encode_text(
                text,
                tokenizer.en_token_to_id,
                tokenizer.en_vocab
            )

            input_tokens = [tokenizer.en_token_to_id['<start>']] + input_tokens

            decoder_input = torch.tensor([input_tokens], dtype=torch.int64).to(device)

            output = model.generate(
                decoder_input,
                max_len=30,
                end_token_id=tokenizer.en_token_to_id['<end>']
            )

            generated_text = tokenizer.decode_ids(
                output[0][1:].cpu().numpy(),
                tokenizer.en_id_to_token
            )

            print(f"Epoch {epoch} - Prompt: {text}, Generation: {generated_text}")
            generations.append([text, generated_text])

    generation_text = f"Epoch {epoch} Generations:\n"
    for prompt, gen in generations:
        generation_text += f"Prompt: {prompt}\nGeneration: {gen}\n\n"
    
    mlflow.log_text(generation_text, f"generations_epoch_{epoch}.txt")
    
    for i, (prompt, gen) in enumerate(generations):
        mlflow.log_metric(f"generation_{i}_prompt", prompt, step=epoch)
        mlflow.log_metric(f"generation_{i}_text", gen, step=epoch)


def get_weights_file_path(cfg: DictConfig, epoch: int, prefix="decoder_pretrain_") -> str:
    model_dir = Path(cfg.data.base_dir) / "decoder_checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir / f"{prefix}{epoch:02d}.pt")


def latest_weights_file_path(cfg: DictConfig, prefix="decoder_pretrain_") -> str:
    model_dir = Path(cfg.data.base_dir) / "decoder_checkpoints"
    if not model_dir.exists():
        return None
    checkpoints = list(model_dir.glob(f"{prefix}*.pt"))
    if not checkpoints:
        return None
    checkpoints.sort()
    return str(checkpoints[-1])


if __name__ == "__main__":
    pretrain_decoder()
