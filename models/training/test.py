import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
import numpy as np
from typing import Dict, List, Tuple
import json

from models.data.dataset import BilingualTranslationDataset, load_hf_dataset
from models.transformer.transformer import TransformerWithNewTechniques
from tokenizer.tokenizer import Tokenizer


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


def create_test_dataset(cfg: DictConfig, tokenizer, dataset):
    print("Creating test dataset from validation split...")
    
    try:
        from datasets import load_dataset
        test_data = load_dataset(
            cfg.dataset.name,
            cfg.dataset.config_name,
            split="test[:1000]"
        )
        test_dataset_dict = {"test": test_data}
    except:
        print("Test split not available, using validation split...")
        val_data = dataset[cfg.dataset.validation_split]
        test_size = min(1000, len(val_data))
        test_data = val_data.select(range(len(val_data) - test_size, len(val_data)))
        test_dataset_dict = {"test": test_data}
    
    test_dataset = BilingualTranslationDataset(
        test_dataset_dict,
        tokenizer,
        cfg.language.src_lang,
        cfg.language.tgt_lang,
        cfg.training.seq_len,
        split="test"
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=getattr(cfg.training, 'num_workers', 0)
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    return test_loader


def load_model_from_checkpoint(cfg: DictConfig, tokenizer, checkpoint_path: str, device):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        global_step = checkpoint.get("global_step", "unknown")
        print(f"Checkpoint loaded: Epoch {epoch}, Step {global_step}")
    else:
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded (no metadata found)")
    
    model.eval()
    return model


def run_test(model, test_loader, device, loss_fn, tokenizer, cfg) -> Dict:
    print("\n" + "="*80)
    print("Starting model testing...")
    print("="*80 + "\n")
    
    model.eval()
    total_loss = 0
    total_bleu = 0
    total_samples = 0
    all_predictions = []
    all_references = []
    all_sources = []
    
    test_iterator = tqdm(
        enumerate(test_loader),
        desc="Testing",
        total=len(test_loader)
    )
    
    with torch.no_grad():
        for batch_idx, batch in test_iterator:
            try:
                inputs = {k: v.to(device) for k, v in batch.items() if k not in ['src_text', 'tgt_text']}
                
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
                    pred = tokenizer.decode_ids(
                        translated[i].cpu().numpy(),
                        tokenizer.en_id_to_token
                    )
                    ref = batch['tgt_text'][i]
                    src = batch['src_text'][i]
                    
                    bleu_score = calculate_bleu(pred, ref) * 100
                    total_bleu += bleu_score
                    total_samples += 1
                    
                    all_predictions.append(pred)
                    all_references.append(ref)
                    all_sources.append(src)
                
                test_iterator.set_postfix(
                    loss=f"{loss:.4f}",
                    bleu=f"{total_bleu/total_samples:.2f}",
                    samples=total_samples
                )
                
            except Exception as e:
                print(f"\nError in test batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    avg_bleu = total_bleu / total_samples if total_samples > 0 else 0
    
    results = {
        "average_loss": avg_loss,
        "average_bleu": avg_bleu,
        "total_samples": total_samples,
        "predictions": all_predictions,
        "references": all_references,
        "sources": all_sources
    }
    
    return results


def print_test_results(results: Dict, cfg: DictConfig, num_examples: int = 10):
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"\nAverage Loss: {results['average_loss']:.4f}")
    print(f"Average BLEU Score: {results['average_bleu']:.2f}")
    print(f"Total Samples: {results['total_samples']}")
    
    print("\n" + "="*80)
    print(f"EXAMPLE TRANSLATIONS (showing {num_examples} examples)")
    print("="*80 + "\n")
    
    num_to_show = min(num_examples, len(results['predictions']))
    indices = np.linspace(0, len(results['predictions']) - 1, num_to_show, dtype=int)
    
    for idx, i in enumerate(indices):
        src = results['sources'][i]
        ref = results['references'][i]
        pred = results['predictions'][i]
        bleu = calculate_bleu(pred, ref) * 100
        
        print(f"Example {idx + 1}:")
        print(f"  Source (RU):     {src}")
        print(f"  Reference (EN):  {ref}")
        print(f"  Prediction (EN): {pred}")
        print(f"  BLEU Score:      {bleu:.2f}")
        print()


def save_test_results(results: Dict, cfg: DictConfig, output_path: str):
    print(f"\nSaving test results to {output_path}...")
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "average_loss": results['average_loss'],
        "average_bleu": results['average_bleu'],
        "total_samples": results['total_samples']
    }
    
    with open(output_path.replace('.json', '_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    predictions_data = []
    for i in range(len(results['predictions'])):
        predictions_data.append({
            "source": results['sources'][i],
            "reference": results['references'][i],
            "prediction": results['predictions'][i],
            "bleu_score": calculate_bleu(results['predictions'][i], results['references'][i]) * 100
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)
    
    report_path = output_path.replace('.json', '_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL TEST REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average Loss: {results['average_loss']:.4f}\n")
        f.write(f"Average BLEU Score: {results['average_bleu']:.2f}\n")
        f.write(f"Total Samples: {results['total_samples']}\n\n")
        f.write("="*80 + "\n")
        f.write("SAMPLE TRANSLATIONS\n")
        f.write("="*80 + "\n\n")
        
        for i, pred_data in enumerate(predictions_data[:20]):
            f.write(f"Example {i + 1}:\n")
            f.write(f"  Source (RU):     {pred_data['source']}\n")
            f.write(f"  Reference (EN):  {pred_data['reference']}\n")
            f.write(f"  Prediction (EN): {pred_data['prediction']}\n")
            f.write(f"  BLEU Score:      {pred_data['bleu_score']:.2f}\n\n")
    
    print(f"Results saved:")
    print(f"  - Metrics: {output_path.replace('.json', '_metrics.json')}")
    print(f"  - Predictions: {output_path}")
    print(f"  - Report: {report_path}")


@hydra.main(config_path="../../models/configs", config_name="config", version_base="1.2")
def test_model(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer({
        'ru_token_to_id': cfg.vocabs.ru_token_to_id,
        'ru_id_to_token': cfg.vocabs.ru_id_to_token,
        'en_token_to_id': cfg.vocabs.en_token_to_id,
        'en_id_to_token': cfg.vocabs.en_id_to_token
    })
    
    print("\nLoading dataset...")
    dataset = load_hf_dataset(cfg)
    
    test_loader = create_test_dataset(cfg, tokenizer, dataset)
    
    checkpoint_path = Path(cfg.data.model_weights)
    if not checkpoint_path.exists():
        alt_checkpoint = Path("checkpoints/transformer_latest.pt")
        if alt_checkpoint.exists():
            checkpoint_path = alt_checkpoint
        else:
            raise FileNotFoundError(f"Checkpoint not found at {cfg.data.model_weights} or checkpoints/transformer_latest.pt")
    
    model = load_model_from_checkpoint(cfg, tokenizer, str(checkpoint_path), device)
    
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.en_token_to_id['<pad>'],
        label_smoothing=0.1
    )
    
    results = run_test(model, test_loader, device, loss_fn, tokenizer, cfg)
    
    print_test_results(results, cfg, num_examples=15)
    
    output_path = Path(cfg.data.log_dir) / "test_results.json"
    save_test_results(results, cfg, str(output_path))
    
    print("\n" + "="*80)
    print("Testing completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_model()

