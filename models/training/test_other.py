import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
from transformers import (
    M2M100ForConditionalGeneration, M2M100Tokenizer,
    MarianMTModel, MarianTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
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


class HuggingFaceModelWrapper:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        print(f"Loading {self.model_name}...")

        try:
            if "m2m100" in self.model_name:
                self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
                self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
                self.src_lang = "ru"
                self.tgt_lang = "en"
                self.model_type = "m2m100"
            elif "opus-mt" in self.model_name:
                self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
                self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)
                self.model_type = "marian"
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
                self.model_type = "auto"

            self.model.eval()
            print(f"✓ {self.model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def translate(self, text: str) -> str:
        try:
            with torch.no_grad():
                if self.model_type == "m2m100":
                    self.tokenizer.src_lang = self.src_lang
                    encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                    generated = self.model.generate(
                        **encoded,
                        forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                        max_length=512
                    )
                    translation = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                else:
                    encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                    generated = self.model.generate(**encoded, max_length=512)
                    translation = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

            return translation
        except Exception as e:
            print(f"Error translating '{text[:50]}...': {e}")
            return ""


def test_hf_model(model_wrapper: HuggingFaceModelWrapper, dataset_split, max_samples: int = None):
    model_wrapper.model.eval()
    total_bleu = 0
    total_samples = 0
    translations_log = []

    if max_samples:
        num_samples = min(len(dataset_split), max_samples)
    else:
        num_samples = len(dataset_split)

    print(f"Testing on {num_samples} samples...")

    for idx in tqdm(range(num_samples), desc=f"Testing {model_wrapper.model_name}", leave=True):
        try:
            example = dataset_split[idx]

            src_text = example.get("ru", "")
            tgt_text = example.get("en", "")

            if not src_text or not tgt_text:
                continue

            translation = model_wrapper.translate(src_text)

            if not translation:
                continue

            bleu_score = calculate_bleu(translation, tgt_text)

            total_bleu += bleu_score
            total_samples += 1

            if len(translations_log) < 10:
                translations_log.append({
                    "source": src_text[:100],
                    "reference": tgt_text[:100],
                    "translation": translation[:100],
                    "bleu": float(bleu_score)
                })

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    avg_bleu = total_bleu / total_samples if total_samples > 0 else 0

    return avg_bleu, total_samples, translations_log


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 80)
    print("Loading OPUS-100 ru-en dataset...")
    try:
        dataset = load_dataset("opus100", "ru-en", split="test", trust_remote_code=True)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading test split: {e}")
        print("Trying validation split...")
        try:
            dataset = load_dataset("opus100", "ru-en", split="validation", trust_remote_code=True)
            print(f"✓ Dataset loaded (validation split): {len(dataset)} samples")
        except Exception as e2:
            print(f"Error loading validation split: {e2}")
            print("Trying to load entire dataset...")
            try:
                dataset = load_dataset("opus100", "ru-en", trust_remote_code=True)
                if isinstance(dataset, dict):
                    dataset = dataset["train"]
                print(f"✓ Dataset loaded (train split): {len(dataset)} samples")
            except Exception as e3:
                print(f"Error loading dataset: {e3}")
                return

    max_samples = 100

    models_to_test = [
        "facebook/m2m100_418M",
        "Helsinki-NLP/opus-mt-ru-en",
        "facebook/m2m100_1.2B"
    ]

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "opus100-ru-en",
        "direction": "Russian to English",
        "max_samples": max_samples,
        "total_dataset_size": len(dataset),
        "models": {}
    }

    for model_name in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"Testing: {model_name}")
        print(f"{'=' * 80}")

        try:
            model_wrapper = HuggingFaceModelWrapper(model_name, device)

            avg_bleu, total_samples, translations_log = test_hf_model(
                model_wrapper, dataset, max_samples=max_samples
            )

            print(f"\n{model_name} Results:")
            print(f"  BLEU Score: {avg_bleu:.4f}")
            print(f"  Samples Tested: {total_samples}")

            all_results["models"][model_name] = {
                "bleu_score": float(avg_bleu),
                "total_samples": int(total_samples),
                "sample_translations": translations_log
            }

            del model_wrapper
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            all_results["models"][model_name] = {
                "error": str(e),
                "bleu_score": None,
                "total_samples": 0
            }

    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = results_dir / f"opus100_ru_en_test_{timestamp_str}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    txt_path = results_dir / f"opus100_ru_en_test_{timestamp_str}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HuggingFace Models Test on OPUS-100 RU-EN Dataset\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {all_results['timestamp']}\n")
        f.write(f"Dataset: {all_results['dataset']}\n")
        f.write(f"Direction: {all_results['direction']}\n")
        f.write(f"Total Dataset Size: {all_results['total_dataset_size']}\n")
        f.write(f"Samples Tested: {all_results['max_samples']}\n\n")
        f.write("Models Tested:\n")
        f.write("-" * 80 + "\n")

        for model_name, results in all_results["models"].items():
            f.write(f"\nModel: {model_name}\n")
            if "error" in results:
                f.write(f"  Error: {results['error']}\n")
            else:
                f.write(f"  BLEU Score: {results['bleu_score']:.4f}\n")
                f.write(f"  Samples: {results['total_samples']}\n")

                if results.get('sample_translations'):
                    f.write(f"  Sample Translations:\n")
                    for i, sample in enumerate(results['sample_translations'], 1):
                        f.write(f"\n    Example {i}:\n")
                        f.write(f"      Source (RU): {sample['source']}\n")
                        f.write(f"      Reference (EN): {sample['reference']}\n")
                        f.write(f"      Translation: {sample['translation']}\n")
                        f.write(f"      BLEU: {sample['bleu']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Summary (sorted by BLEU score):\n")
        f.write("-" * 80 + "\n")

        bleu_scores = {}
        for model_name, results in all_results["models"].items():
            if results.get('bleu_score') is not None:
                bleu_scores[model_name] = results['bleu_score']

        for model_name, bleu in sorted(bleu_scores.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{model_name}: {bleu:.4f}\n")

        f.write("=" * 80 + "\n")

    print(f"\n✓ Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")


if __name__ == "__main__":
    main()