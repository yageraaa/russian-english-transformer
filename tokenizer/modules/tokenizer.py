import json
import re

class Tokenizer:
    def __init__(self, vocab_paths=None, special_tokens=None):
        self.vocabs = {}
        self.special_tokens = special_tokens or ['<start>', '<end>', '<unk>', '<pad>']
        self.vocab_paths = vocab_paths or {}
        self.load_vocabularies()

    def detect_language(self, text):
        return 'ru' if re.search(r'[а-яёА-ЯЁ]', text) else 'en'

    def clean_text(self, text, lang):
        pattern = r"[^а-яё\s,.!?0-9'-]" if lang == 'ru' else r"[^a-z\s,.!?0-9'-]"
        text = re.sub(pattern, "", text.lower())
        return ' '.join(text.split())

    def load_vocabularies(self):
        for lang in ['ru', 'en']:
            token_to_id_path = self.vocab_paths.get(f'{lang}_token_to_id')
            id_to_token_path = self.vocab_paths.get(f'{lang}_id_to_token')

            if token_to_id_path and id_to_token_path:
                with open(token_to_id_path, 'r', encoding='utf-8') as f:
                    setattr(self, f'{lang}_token_to_id', json.load(f))
                with open(id_to_token_path, 'r', encoding='utf-8') as f:
                    id_to_token = json.load(f)
                    setattr(self, f'{lang}_id_to_token', {int(k): v for k, v in id_to_token.items()})
                    setattr(self, f'{lang}_vocab', set(id_to_token.values()))
            else:
                print(f"Warning: Missing vocab for {lang}.")

    def tokenize_word(self, word, vocab):
        symbols = list(word)
        i, tokens = 0, []
        while i < len(symbols):
            for j in range(len(symbols), i, -1):
                substr = ''.join(symbols[i:j])
                if substr in vocab:
                    tokens.append(substr)
                    i = j
                    break
            else:
                tokens.append('<unk>')
                i += 1
        return tokens

    def encode_text(self, text, token_to_id, vocab):
        tokens = ['<start>']
        for word in text.split():
            tokens.extend(self.tokenize_word(word, vocab))
        tokens.append('<end>')
        return [token_to_id.get(token, token_to_id['<unk>']) for token in tokens]

    def decode_ids(self, token_ids, id_to_token):
        tokens = [id_to_token.get(i, '<unk>') for i in token_ids]
        words = []
        for token in tokens:
            if token in ['<start>', '<end>', '<pad>']:
                continue
            if token in ['.', ',', '?', '!', '-', "'"]:
                if words:
                    words[-1] += token
                else:
                    words.append(token)
            else:
                words.append(token)
        return ' '.join(words).strip().capitalize()
