import json
import re

class Tokenizer:
    def __init__(self, vocab_paths=None, special_tokens=None):
        self.vocabs = {}
        self.special_tokens = special_tokens or ['<start>', '<end>', '<unk>', '<pad>']
        self.vocab_paths = vocab_paths or {}
        self.load_vocabularies()

    def detect_language(self, text):
        if re.search(r'[а-яёА-ЯЁ]', text):
            return 'ru'
        else:
            return 'en'

    def clean_text(self, text, lang):
        if lang == 'ru':
            pattern = r"[^а-яё\s]"
        else:
            pattern = r"[^a-z\s]"

        text = re.sub(pattern, "", text.lower())
        return ' '.join(text.split())

    def load_vocabularies(self):
        language_codes = {'ru', 'en'}
        for lang in language_codes:
            token_to_id_path = self.vocab_paths.get(f'{lang}_token_to_id')
            id_to_token_path = self.vocab_paths.get(f'{lang}_id_to_token')

            if token_to_id_path and id_to_token_path:
                with open(token_to_id_path, 'r', encoding='utf-8') as f:
                    setattr(self, f'{lang}_token_to_id', json.load(f))
                with open(id_to_token_path, 'r', encoding='utf-8') as f:
                    id_to_token = json.load(f)
                    id_to_token = {int(k): v for k, v in id_to_token.items()}
                    setattr(self, f'{lang}_id_to_token', id_to_token)

                setattr(self, f'{lang}_vocab', set(id_to_token.values()))

                print(f"Загружены словари для языка {lang}.")

            else:
                print(f"Предупреждение: Пути к словарям для языка {lang} не указаны.")

    def tokenize_word(self, word, vocab):
        symbols = list(word) + ['_ed']
        i = 0
        tokens = []
        while i < len(symbols):
            j = len(symbols)
            found = False
            while j > i:
                substring = ''.join(symbols[i:j])
                if substring in vocab:
                    tokens.append(substring)
                    i = j
                    found = True
                    break
                else:
                    j -= 1
            if not found:
                tokens.append('<unk>')
                i += 1
        return tokens

    def encode_text(self, text, token_to_id, vocab):
        token_ids = []
        for word in text.split():
            word_tokens = self.tokenize_word(word, vocab)
            for token in word_tokens:
                token_id = token_to_id.get(token)
                if token_id is not None:
                    token_ids.append(token_id)
                else:
                    token_ids.append(token_to_id.get('<unk>'))
        return token_ids

    def decode_ids(self, token_ids, id_to_token):
        tokens = [id_to_token.get(token_id, '<unk>') for token_id in token_ids]
        words = []
        word = ''
        for token in tokens:
            if token == '_ed':
                words.append(word)
                word = ''
            elif token not in ['<start>', '<end>', '<pad>']:
                word += token
        if word:
            words.append(word)
        return ' '.join(words)





