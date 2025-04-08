import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from time import time

from models.modules.configuration import get_config, get_weights_file_path, latest_weights_file_path
from models.modules.dataset import BilingualTranslationDataset, load_local_dataset
from models.modules.transformer import build_transformer
from tokenizer.modules.tokenizer import Tokenizer


def train_model():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="transformer-ru-en",
        name=f"{config['experiment_name']}-{int(time())}",
        config=config
    )

    tokenizer = Tokenizer(config["vocab_paths"])
    train_ds, val_ds = create_datasets(config, tokenizer)

    model = build_transformer(
        src_vocab_size=len(tokenizer.ru_token_to_id),
        tgt_vocab_size=len(tokenizer.en_token_to_id),
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        d_ff=config["d_ff"]
    ).to(device)

    wandb.watch(model, log="all", log_freq=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.ru_token_to_id['<pad>'], label_smoothing=0.1)

    epoch, global_step = load_checkpoint(config, model, optimizer)

    for epoch in range(epoch, config["num_epochs"]):
        model.train()
        progress_bar = tqdm(train_ds, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")

        for batch in progress_bar:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'src_text' and k != 'tgt_text'}
            encoder_output = model.encode(inputs['encoder_input'], inputs['encoder_mask'])
            decoder_output = model.decode(encoder_output, inputs['encoder_mask'], inputs['decoder_input'], inputs['decoder_mask'])
            proj_output = model.project(decoder_output)
            loss = loss_fn(proj_output.view(-1, len(tokenizer.en_token_to_id)), inputs['label'].view(-1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if global_step % config["log_interval"] == 0:
                log_data = {
                    "train/loss": loss.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "step": global_step
                }

                if config["log_examples"] and global_step % 500 == 0:
                    log_data.update(log_translations(model, tokenizer, device, config))

                wandb.log(log_data)

            global_step += 1
            progress_bar.set_postfix(loss=f"{loss.item():.3f}")

        val_loss = run_validation(model, val_ds, device, loss_fn)
        wandb.log({"val/loss": val_loss, "epoch": epoch})
        save_checkpoint(config, epoch, global_step, model, optimizer)

    wandb.finish()


def create_datasets(config, tokenizer):
    train_data = load_local_dataset(config["dataset_path"]["train_ru"], config["dataset_path"]["train_en"])
    train, val = random_split(train_data, [int(0.9 * len(train_data)), len(train_data) - int(0.9 * len(train_data))])

    return (
        DataLoader(
            BilingualTranslationDataset(train, tokenizer, config["src_lang"], config["tgt_lang"], config["seq_len"]),
            batch_size=config["batch_size"], shuffle=True, pin_memory=True),
        DataLoader(
            BilingualTranslationDataset(val, tokenizer, config["src_lang"], config["tgt_lang"], config["seq_len"]),
            batch_size=config["batch_size"], pin_memory=True)
    )


def load_checkpoint(config, model, optimizer):
    if config["preload"] == "latest":
        if model_file := latest_weights_file_path(config):
            state = torch.load(model_file)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            return state["epoch"] + 1, state["global_step"]
    return 0, 0


def save_checkpoint(config, epoch, step, model, optimizer):
    model_file = get_weights_file_path(config, epoch)
    torch.save({
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, model_file)
    wandb.log_artifact(wandb.Artifact(f"model-epoch-{epoch}", type="model", metadata=wandb.config).add_file(model_file))


def run_validation(model, val_loader, device, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'src_text' and k != 'tgt_text'}
            encoder_output = model.encode(inputs['encoder_input'], inputs['encoder_mask'])
            decoder_output = model.decode(encoder_output, inputs['encoder_mask'], inputs['decoder_input'], inputs['decoder_mask'])
            proj_output = model.project(decoder_output)
            total_loss += loss_fn(proj_output.view(-1, inputs['label'].size(-1)), inputs['label'].view(-1)).item()

    return total_loss / len(val_loader)


def log_translations(model, tokenizer, device, config):
    examples = [
        ("Привет, как дела?", "Hello, how are you?"),
        ("Сегодня хорошая погода", "The weather is nice today")
    ]

    model.eval()
    translations = []

    with torch.no_grad():
        for src, _ in examples:
            input_tokens = tokenizer.encode_text(
                src,
                getattr(tokenizer, f"{config['src_lang']}_token_to_id"),
                getattr(tokenizer, f"{config['src_lang']}_vocab")
            )
            encoder_input = torch.tensor([input_tokens], dtype=torch.int64).to(device)
            output = model.translate(encoder_input)
            translations.append([src, tokenizer.decode(output[0].cpu().numpy(), config["tgt_lang"])])

    return {"examples": wandb.Table(columns=["Source", "Translation"], data=translations)}


if __name__ == "__main__":
    train_model()