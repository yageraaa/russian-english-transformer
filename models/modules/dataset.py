import torch
from torch.utils.data import Dataset
from tokenizer.modules.tokenizer import Tokenizer
from pathlib import Path


class BilingualTranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, src_lang='ru', tgt_lang='en', seq_length=128):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_length = seq_length
        src_token_to_id = getattr(tokenizer, f'{src_lang}_token_to_id')
        self.sos_token = torch.tensor([src_token_to_id['<start>']], dtype=torch.int64)
        self.eos_token = torch.tensor([src_token_to_id['<end>']], dtype=torch.int64)
        self.pad_token = torch.tensor([src_token_to_id['<pad>']], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]
        src_clean = self.tokenizer.clean_text(src_text, self.src_lang)
        tgt_clean = self.tokenizer.clean_text(tgt_text, self.tgt_lang)
        src_token_to_id = getattr(self.tokenizer, f'{self.src_lang}_token_to_id')
        src_vocab = getattr(self.tokenizer, f'{self.src_lang}_vocab')
        tgt_token_to_id = getattr(self.tokenizer, f'{self.tgt_lang}_token_to_id')
        tgt_vocab = getattr(self.tokenizer, f'{self.tgt_lang}_vocab')
        src_ids = self.tokenizer.encode_text(src_clean, src_token_to_id, src_vocab)
        tgt_ids = self.tokenizer.encode_text(tgt_clean, tgt_token_to_id, tgt_vocab)
        encoder_input = self._add_special_tokens_and_pad(src_ids, add_eos=True)
        decoder_input = self._add_special_tokens_and_pad(tgt_ids, add_eos=False)
        label = self._add_special_tokens_and_pad(tgt_ids, add_eos=True, add_sos=False)
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).int()
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & \
                       self.causal_mask(decoder_input.size(0))

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
        return torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) == 0


def load_local_dataset(ru_path, en_path):
    with open(ru_path, 'r', encoding='utf-8') as f_ru, open(en_path, 'r', encoding='utf-8') as f_en:
        ru_lines = [line.strip() for line in f_ru.readlines()]
        en_lines = [line.strip() for line in f_en.readlines()]

    return [{'ru': ru, 'en': en} for ru, en in zip(ru_lines, en_lines)]


if __name__ == "__main__":
    BASE_DIR = Path('/home/gera/PycharmProjects/russian-english-transformer')
    ru_path = BASE_DIR / 'models/data/ru.txt'
    en_path = BASE_DIR / 'models/data/en.txt'

    dataset = load_local_dataset(ru_path, en_path)

    tokenizer = Tokenizer(
        vocab_paths={
            'ru_token_to_id': str(BASE_DIR / 'tokenizer/vocabs/ru-vocab/ru_token_to_id.json'),
            'ru_id_to_token': str(BASE_DIR / 'tokenizer/vocabs/ru-vocab/ru_id_to_token.json'),
            'en_token_to_id': str(BASE_DIR / 'tokenizer/vocabs/en-vocab/en_token_to_id.json'),
            'en_id_to_token': str(BASE_DIR / 'tokenizer/vocabs/en-vocab/en_id_to_token.json')
        }
    )

    train_dataset = BilingualTranslationDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        src_lang='ru',
        tgt_lang='en',
        seq_length=350
    )

    sample = train_dataset[0]
    print("\nПример преобразования:")
    print(f"Исходный текст (ru): {sample['src_text']}")
    print(f"Целевой текст (en): {sample['tgt_text']}")
    print(f"\nФорма encoder_input: {sample['encoder_input'].shape}")
    print(f"Форма decoder_mask: {sample['decoder_mask'].shape}")