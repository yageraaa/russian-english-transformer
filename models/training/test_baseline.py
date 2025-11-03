import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from accelerate import Accelerator
from models.data.dataset import BilingualTranslationDataset, load_hf_dataset
from datasets import load_dataset
from models.transformer.transformer import TransformerBaseline
from tokenizer.tokenizer import Tokenizer
import re
import json
from datetime import datetime


def decode_until_end(ids, id_to_token, end_token="<end>"):
    tokens = []
    for idx in ids:
        token = id_to_token.get(idx, '<unk>')
        if token == end_token:
            break
        tokens.append(token)
    return " ".join(tokens)


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


def load_model_from_checkpoint(cfg: DictConfig, tokenizer, checkpoint_path: str, device):
    print(f"Loading baseline model...")
    
    model = TransformerBaseline(
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
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"  - {key}")
        else:
            for key in missing_keys[:10]:
                print(f"  - {key}")
            print(f"  ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"  - {key}")
        else:
            for key in unexpected_keys[:10]:
                print(f"  - {key}")
            print(f"  ... and {len(unexpected_keys) - 10} more")
    
    model.eval()
    print("✓ Baseline model loaded successfully")
    return model


def run_test(unwrapped_model, test_loader, device, loss_fn, tokenizer, cfg, accelerator):
    accelerator.wait_for_everyone()
    
    unwrapped_model.eval()
    total_loss = 0
    total_bleu = 0
    total_samples = 0
    
    test_iterator = tqdm(
        enumerate(test_loader),
        desc="Testing",
        total=len(test_loader),
        leave=False,
        disable=not accelerator.is_local_main_process
    )
    
    with torch.no_grad():
        for batch_idx, batch in test_iterator:
            try:
                inputs = {k: v for k, v in batch.items() if k != 'src_text' and k != 'tgt_text'}
                
                with torch.amp.autocast('cuda',
                                        dtype=torch.bfloat16 if cfg.training.mixed_precision == 'bf16' else torch.float16 if cfg.training.mixed_precision == 'fp16' else torch.float32):
                    encoder_output = unwrapped_model.encode(inputs['encoder_input'], inputs['encoder_mask'])
                    decoder_output = unwrapped_model.decode(
                        encoder_output,
                        inputs['encoder_mask'],
                        inputs['decoder_input'],
                        inputs['decoder_mask']
                    )
                    proj_output = unwrapped_model.project(decoder_output)
                    
                    loss = loss_fn(
                        proj_output.view(-1, len(tokenizer.en_token_to_id)),
                        inputs['label'].view(-1)
                    ).item()
                
                total_loss += loss
                
                translated = unwrapped_model.translate_batch(
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
                
                test_iterator.set_postfix(loss=f"{loss:.4f}", bleu=f"{total_bleu/total_samples:.4f}", samples=total_samples)
            
            except Exception as e:
                accelerator.print(f"Error in test batch {batch_idx}: {e}")
                accelerator.print("Skipping this test batch...")
                continue
    
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    avg_bleu = total_bleu / total_samples if total_samples > 0 else 0
    
    accelerator.print(f"Test results: Loss = {avg_loss:.4f}, BLEU = {avg_bleu:.4f}, Samples = {total_samples}")
    
    return avg_loss, avg_bleu, total_samples


def load_test_dataset(cfg: DictConfig, accelerator=None):
    test_dataset_name = getattr(cfg.dataset, 'test_dataset_name', 'opus100')
    test_config_name = getattr(cfg.dataset, 'test_config_name', 'ru-en')
    test_split = getattr(cfg.dataset, 'test_split', 'test')
    
    print_func = accelerator.print if accelerator else print
    print_func(f"Loading test dataset: {test_dataset_name}/{test_config_name} (split: {test_split})...")
    
    try:
        dataset_dict = load_dataset(test_dataset_name, test_config_name, split=test_split)
        
        dataset = {test_split: dataset_dict}
        
        print_func(f"Test dataset loaded: {len(dataset_dict)} examples")
        
        if len(dataset_dict) > 0:
            sample = dataset_dict[0]
            print_func("\nSample translation:")
            print_func(f"Source ({cfg.language.src_lang}): {sample['translation'][cfg.language.src_lang]}")
            print_func(f"Target ({cfg.language.tgt_lang}): {sample['translation'][cfg.language.tgt_lang]}")
        
        return dataset
    except Exception as e:
        print_func(f"Error loading test dataset: {e}")
        raise


def create_test_dataset(cfg: DictConfig, tokenizer, dataset):
    test_split = getattr(cfg.dataset, 'test_split', 'test')
    
    test_dataset = BilingualTranslationDataset(
        dataset, tokenizer, cfg.language.src_lang, cfg.language.tgt_lang, cfg.training.seq_len,
        split=test_split
    )
    
    return DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        pin_memory=True,
        num_workers=getattr(cfg.training, 'num_workers', 0)
    )


@hydra.main(config_path="../../models/configs", config_name="config", version_base="1.2")
def test_model(cfg: DictConfig):
    accelerator = Accelerator()
    device = accelerator.device
    
    accelerator.print("Loading tokenizer...")
    tokenizer = Tokenizer({
        'ru_token_to_id': cfg.vocabs.ru_token_to_id,
        'ru_id_to_token': cfg.vocabs.ru_id_to_token,
        'en_token_to_id': cfg.vocabs.en_token_to_id,
        'en_id_to_token': cfg.vocabs.en_id_to_token
    })
    
    accelerator.print("Loading test dataset from HuggingFace...")
    test_dataset = load_test_dataset(cfg, tokenizer)

    possible_paths = [
        Path("checkpoints_baseline/transformer_latest.pt"),
        Path(cfg.data.base_dir) / "checkpoints_baseline" / "transformer_latest.pt",
        Path(cfg.data.model_dir) / "transformer_latest.pt",
        Path("checkpoints") / "transformer_latest.pt",
    ]
    
    baseline_checkpoint_path = None
    for path in possible_paths:
        if path.exists():
            baseline_checkpoint_path = path
            break
    
    if baseline_checkpoint_path is None:
        accelerator.print("Error: Baseline checkpoint not found. Tried:")
        for path in possible_paths:
            accelerator.print(f"  - {path.resolve()}")
        accelerator.print("Please ensure the baseline checkpoint exists at one of these locations.")
        return
    
    accelerator.print(f"Found baseline checkpoint at: {baseline_checkpoint_path.resolve()}")
    
    accelerator.print(f"Loading model from checkpoint: {baseline_checkpoint_path}")
    model = load_model_from_checkpoint(cfg, tokenizer, str(baseline_checkpoint_path), device)
    
    accelerator.print("Creating test dataset...")
    test_ds = create_test_dataset(cfg, tokenizer, test_dataset)
    accelerator.print(f"Test dataset size: {len(test_ds.dataset)}")
    
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.en_token_to_id['<pad>'],
        label_smoothing=0.1
    )
    
    model, test_ds = accelerator.prepare(model, test_ds)
    
    unwrapped_model = accelerator.unwrap_model(model)
    
    accelerator.print("\n" + "="*80)
    accelerator.print("Starting test evaluation...")
    accelerator.print("="*80 + "\n")
    
    avg_loss, avg_bleu, total_samples = run_test(unwrapped_model, test_ds, device, loss_fn, tokenizer, cfg, accelerator)
    
    accelerator.print("\n" + "="*80)
    accelerator.print("Test Summary:")
    accelerator.print(f"  Average Loss: {avg_loss:.4f}")
    accelerator.print(f"  Average BLEU: {avg_bleu:.4f}")
    accelerator.print(f"  Total Samples: {total_samples}")
    accelerator.print("="*80)
    
    if accelerator.is_local_main_process:
        results = {
            "checkpoint_path": str(baseline_checkpoint_path),
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "average_loss": float(avg_loss),
                "average_bleu": float(avg_bleu),
                "total_samples": int(total_samples)
            },
            "model_config": {
                "d_model": cfg.model.d_model,
                "num_layers": cfg.model.num_layers,
                "num_heads": cfg.model.num_heads,
                "d_ff": cfg.model.d_ff,
                "dropout": cfg.training.dropout,
                "seq_len": cfg.training.seq_len
            }
        }
        
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = results_dir / f"baseline_test_results_{timestamp_str}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        txt_path = results_dir / f"baseline_test_results_{timestamp_str}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Baseline Transformer Test Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Checkpoint: {baseline_checkpoint_path}\n\n")
            f.write("Metrics:\n")
            f.write(f"  Average Loss: {avg_loss:.4f}\n")
            f.write(f"  Average BLEU: {avg_bleu:.4f}\n")
            f.write(f"  Total Samples: {total_samples}\n\n")
            f.write("Model Configuration:\n")
            f.write(f"  d_model: {cfg.model.d_model}\n")
            f.write(f"  num_layers: {cfg.model.num_layers}\n")
            f.write(f"  num_heads: {cfg.model.num_heads}\n")
            f.write(f"  d_ff: {cfg.model.d_ff}\n")
            f.write(f"  dropout: {cfg.training.dropout}\n")
            f.write(f"  seq_len: {cfg.training.seq_len}\n")
            f.write("=" * 80 + "\n")
        
        accelerator.print(f"\n✓ Results saved to:")
        accelerator.print(f"  JSON: {json_path}")
        accelerator.print(f"  TXT:  {txt_path}")


if __name__ == "__main__":
    test_model()

