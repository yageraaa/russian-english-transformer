import torch
from torch.utils.data import Dataset
from tokenizer.tokenizer import Tokenizer
from pathlib import Path


class EnglishLanguageModelDataset(Dataset):
    def __init__(self, dataset, tokenizer, tgt_lang='en', seq_length=128):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tgt_lang = tgt_lang
        self.seq_length = seq_length
        tgt_token_to_id = getattr(tokenizer, f'{tgt_lang}_token_to_id')
        self.sos_token = torch.tensor([tgt_token_to_id['<start>']], dtype=torch.int64)
        self.eos_token = torch.tensor([tgt_token_to_id['<end>']], dtype=torch.int64)
        self.pad_token = torch.tensor([tgt_token_to_id['<pad>']], dtype=torch.int64)
        self.tgt_token_to_id = tgt_token_to_id
        self.tgt_vocab = getattr(tokenizer, f'{tgt_lang}_vocab')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        tgt_text = item[self.tgt_lang]
        tgt_clean = self.tokenizer.clean_text(tgt_text, self.tgt_lang)
        tgt_ids = self.tokenizer.encode_text(tgt_clean, self.tgt_token_to_id, self.tgt_vocab)
        decoder_input = self._add_special_tokens_and_pad(tgt_ids, add_eos=False)
        label = self._add_special_tokens_and_pad(tgt_ids, add_eos=True, add_sos=False)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & \
                       self.causal_mask(decoder_input.size(0))

        return {
            "decoder_input": decoder_input,
            "decoder_mask": decoder_mask,
            "label": label,
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


def load_english_dataset(en_path):
    with open(en_path, 'r', encoding='utf-8') as f_en:
        en_lines = [line.strip() for line in f_en.readlines()]

    return [{'en': en} for en in en_lines]


if __name__ == "__main__":
    BASE_DIR = Path('/')
    en_path = BASE_DIR / 'models/data/Tatoeba.en-ru.en'

    dataset = load_english_dataset(en_path)

    tokenizer = Tokenizer(
        vocab_paths={
            'en_token_to_id': str(BASE_DIR / 'tokenizer/vocabs/en-vocab/en_token_to_id.json'),
            'en_id_to_token': str(BASE_DIR / 'tokenizer/vocabs/en-vocab/en_id_to_token.json')
        }
    )

    train_dataset = EnglishLanguageModelDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        tgt_lang='en',
        seq_length=128
    )

    sample = train_dataset[0]
    print("\nПример преобразования:")
    print(f"Текст (en): {sample['tgt_text']}")
    print(f"\nФорма decoder_input: {sample['decoder_input'].shape}")
    print(f"Форма decoder_mask: {sample['decoder_mask'].shape}")
