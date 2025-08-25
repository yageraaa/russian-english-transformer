import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizer.tokenizer import Tokenizer
from tqdm import tqdm
from omegaconf import OmegaConf

class BilingualTranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, src_lang='ru', tgt_lang='en', seq_length=128, split='train'):
        super().__init__()
        self.dataset = dataset[split]
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_length = seq_length
        self.src_token_to_id = getattr(tokenizer, f'{src_lang}_token_to_id')
        self.tgt_token_to_id = getattr(tokenizer, f'{tgt_lang}_token_to_id')
        self.src_vocab = getattr(tokenizer, f'{src_lang}_vocab')
        self.tgt_vocab = getattr(tokenizer, f'{tgt_lang}_vocab')
        self.sos_token = torch.tensor([self.src_token_to_id['<start>']], dtype=torch.int64)
        self.eos_token = torch.tensor([self.src_token_to_id['<end>']], dtype=torch.int64)
        self.pad_token = torch.tensor([self.src_token_to_id['<pad>']], dtype=torch.int64)

        if len(self.dataset) > 0:
            first_item = self.dataset[0]
            if 'translation' not in first_item:
                raise ValueError(
                    f"Dataset does not contain 'translation' field. Available fields: {list(first_item.keys())}")

            translation = first_item['translation']
            if src_lang not in translation or tgt_lang not in translation:
                raise ValueError(
                    f"Translation does not contain required languages. Available: {list(translation.keys())}, Required: {src_lang}, {tgt_lang}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]['translation']
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]

        if not src_text or not tgt_text:
            if idx + 1 < len(self):
                return self[idx + 1]
            else:
                src_text = "Пример"
                tgt_text = "Example"

        src_clean = self.tokenizer.clean_text(src_text, self.src_lang)
        tgt_clean = self.tokenizer.clean_text(tgt_text, self.tgt_lang)
        src_ids = self.tokenizer.encode_text(src_clean, self.src_token_to_id, self.src_vocab)
        tgt_ids = self.tokenizer.encode_text(tgt_clean, self.tgt_token_to_id, self.tgt_vocab)
        encoder_input = self._add_special_tokens_and_pad(src_ids, add_eos=True)
        decoder_input = self._add_special_tokens_and_pad(tgt_ids, add_eos=False)
        label = self._add_special_tokens_and_pad(tgt_ids, add_eos=True, add_sos=False)
        encoder_mask = (encoder_input != self.pad_token).int()  # [seq_len]
        decoder_mask = (decoder_input != self.pad_token).int().unsqueeze(0) & self.causal_mask(decoder_input.size(0))  # [1, seq_len] & [seq_len, seq_len] -> [seq_len, seq_len]

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

    def _add_special_tokens_and_pad(self, token_ids, add_sos=True, add_eos=True):
        tokens = []

        if add_sos:
            tokens.append(self.sos_token)

        tokens.append(torch.tensor(token_ids, dtype=torch.int64))

        if add_eos:
            tokens.append(self.eos_token)

        tensor = torch.cat(tokens, dim=0)

        padding = self.seq_length - tensor.size(0)
        if padding > 0:
            tensor = torch.cat([tensor, torch.full((padding,), self.pad_token.item(), dtype=torch.int64)])
        elif padding < 0:
            tensor = tensor[:self.seq_length]

        return tensor

    @staticmethod
    def causal_mask(size):
        return torch.tril(torch.ones(size, size)).bool()


def load_hf_dataset(cfg):
    print(f"Loading dataset {cfg.dataset.name}/{cfg.dataset.config_name}...")

    try:
        dataset = load_dataset(
            cfg.dataset.name,
            cfg.dataset.config_name,
            split=[cfg.dataset.train_split, cfg.dataset.validation_split]
        )

        result = {
            cfg.dataset.train_split: dataset[0],
            cfg.dataset.validation_split: dataset[1]
        }

        print(f"Train split: {len(result[cfg.dataset.train_split])} examples")
        print(f"Validation split: {len(result[cfg.dataset.validation_split])} examples")

        if len(result[cfg.dataset.train_split]) > 0:
            sample = result[cfg.dataset.train_split][0]
            print("\nSample translation:")
            print(f"Source ({cfg.language.src_lang}): {sample['translation'][cfg.language.src_lang]}")
            print(f"Target ({cfg.language.tgt_lang}): {sample['translation'][cfg.language.tgt_lang]}")

        return result

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


if __name__ == "__main__":

    cfg = OmegaConf.load('../configs/config.yaml')

    tokenizer = Tokenizer(
        vocab_paths={
            'ru_token_to_id': cfg.vocabs.ru_token_to_id,
            'ru_id_to_token': cfg.vocabs.ru_id_to_token,
            'en_token_to_id': cfg.vocabs.en_token_to_id,
            'en_id_to_token': cfg.vocabs.en_id_to_token
        }
    )

    dataset = load_hf_dataset(cfg)
    train_dataset = BilingualTranslationDataset(dataset, tokenizer, split=cfg.dataset.train_split)

    print("\nTesting dataset iteration...")
    for i in tqdm(range(min(5, len(train_dataset)))):
        sample = train_dataset[i]
        print(f"\nSample {i}:")
        print(f"Source: {sample['src_text']}")
        print(f"Target: {sample['tgt_text']}")
        print(f"Encoder input shape: {sample['encoder_input'].shape}")
        print(f"Decoder input shape: {sample['decoder_input'].shape}")
        print(f"Encoder mask shape: {sample['encoder_mask'].shape}")
        print(f"Decoder mask shape: {sample['decoder_mask'].shape}")
        print(f"Label shape: {sample['label'].shape}")