import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import json
from datetime import datetime
from pathlib import Path


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


class T5TranslatorWrapper:
    def __init__(self, model_name: str = "utrobinmv/t5_translate_en_ru_zh_large_1024",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        print(f"Loading {self.model_name}...")

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print(f"✓ {self.model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def translate(self, text: str, source_lang: str = "ru", target_lang: str = "en") -> str:
        try:
            task_prefix = f"translate {source_lang} to {target_lang}: "
            input_text = task_prefix + text

            encoded = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **encoded,
                    max_length=512,
                    early_stopping=True,
                    num_beams=4
                )

            translation = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return translation.strip()
        except Exception as e:
            return ""


def test_t5_model(model_wrapper: T5TranslatorWrapper, dataset_split, max_samples: int = None):
    model_wrapper.model.eval()
    total_bleu = 0
    total_samples = 0
    translations_log = []

    if max_samples:
        num_samples = min(len(dataset_split), max_samples)
    else:
        num_samples = len(dataset_split)

    print(f"Testing on {num_samples} samples...")

    for idx in tqdm(range(num_samples), desc="Testing T5 Model", leave=True):
        try:
            example = dataset_split[idx]

            if isinstance(example, dict) and "translation" in example:
                src_text = example["translation"].get("en", "")
                tgt_text = example["translation"].get("ru", "")
            elif isinstance(example, dict) and "en" in example and "ru" in example:
                src_text = example["en"]
                tgt_text = example["ru"]
            else:
                continue

            if not src_text or not tgt_text:
                continue

            if isinstance(src_text, list):
                src_text = src_text[0] if src_text else ""
            if isinstance(tgt_text, list):
                tgt_text = tgt_text[0] if tgt_text else ""

            if not src_text or not tgt_text:
                continue

            translation = model_wrapper.translate(src_text, source_lang="en", target_lang="ru")

            if not translation:
                continue

            bleu_score = calculate_bleu(translation, tgt_text)

            total_bleu += bleu_score
            total_samples += 1

            if len(translations_log) < 10:
                translations_log.append({
                    "source": str(src_text)[:100],
                    "reference": str(tgt_text)[:100],
                    "translation": str(translation)[:100],
                    "bleu": float(bleu_score)
                })

        except Exception as e:
            continue

    avg_bleu = total_bleu / total_samples if total_samples > 0 else 0

    return avg_bleu, total_samples, translations_log


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 80)

    print("Loading OPUS-100 en-ru dataset...")
    try:
        dataset = load_dataset("opus100", "en-ru", split="train")
        print(f"✓ Dataset loaded: {len(dataset)} samples")

        example = dataset[0]
        print(f"Dataset structure: {example.keys()}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    max_samples = 100

    print(f"\n{'=' * 80}")
    print(f"Testing: utrobinmv/t5_translate_en_ru_zh_large_1024")
    print(f"Translation direction: EN → RU")
    print(f"{'=' * 80}")

    try:
        model_wrapper = T5TranslatorWrapper(device=device)

        avg_bleu, total_samples, translations_log = test_t5_model(
            model_wrapper, dataset, max_samples=max_samples
        )

        print(f"\nT5 Model Results:")
        print(f"  BLEU Score: {avg_bleu:.4f}")
        print(f"  Samples Tested: {total_samples}")

        if translations_log:
            print(f"\nSample translations:")
            for i, sample in enumerate(translations_log[:3], 1):
                print(f"\n  Example {i}:")
                print(f"    Source: {sample['source']}")
                print(f"    Reference: {sample['reference']}")
                print(f"    Translation: {sample['translation']}")
                print(f"    BLEU: {sample['bleu']:.4f}")

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "model": "utrobinmv/t5_translate_en_ru_zh_large_1024",
            "dataset": "opus100-en-ru",
            "direction": "English to Russian",
            "max_samples": max_samples,
            "total_dataset_size": len(dataset),
            "bleu_score": float(avg_bleu),
            "total_samples": int(total_samples),
            "sample_translations": translations_log
        }

        del model_wrapper
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error testing model: {e}")
        all_results = {
            "error": str(e),
            "bleu_score": None,
            "total_samples": 0
        }

    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = results_dir / f"t5_translate_test_{timestamp_str}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    txt_path = results_dir / f"t5_translate_test_{timestamp_str}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("T5 Translation Model Test\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: utrobinmv/t5_translate_en_ru_zh_large_1024\n")
        f.write(f"Timestamp: {all_results.get('timestamp', 'N/A')}\n")
        f.write(f"Dataset: {all_results.get('dataset', 'N/A')}\n")
        f.write(f"Direction: {all_results.get('direction', 'N/A')}\n")
        f.write(f"Samples Tested: {all_results.get('total_samples', 0)}\n\n")

        if "error" not in all_results:
            f.write(f"BLEU Score: {all_results.get('bleu_score', 0):.4f}\n\n")

            if all_results.get('sample_translations'):
                f.write(f"Sample Translations:\n")
                f.write("-" * 80 + "\n")
                for i, sample in enumerate(all_results['sample_translations'], 1):
                    f.write(f"\nExample {i}:\n")
                    f.write(f"  Source (EN): {sample['source']}\n")
                    f.write(f"  Reference (RU): {sample['reference']}\n")
                    f.write(f"  Translation: {sample['translation']}\n")
                    f.write(f"  BLEU: {sample['bleu']:.4f}\n")
        else:
            f.write(f"Error: {all_results.get('error', 'Unknown error')}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\n✓ Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")


if __name__ == "__main__":
    main()