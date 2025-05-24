import torch
import torch.nn as nn
from torchinfo import summary
from models.modules.embeddings import InputEmbeddings
from models.modules.positional_encoding import PositionalEncoding
from models.modules.linear_layer import ProjectionLayer
from models.modules.rms_norm import RMSNorm
from models.modules.swiglu import SwiGLU
from models.modules.qk_norm import QKNorm
from models.modules.decoder_v2 import DecoderWithNewTechniques
from models.modules.encoder_v2 import EncoderWithNewTechniques


class TransformerWithNewTechniques(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len,
                 d_model=512, num_layers=6, num_heads=8, dropout=0.1, d_ff=2048):
        super().__init__()
        self.src_embed = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        self.src_pos = PositionalEncoding(d_model, src_seq_len)
        self.tgt_pos = PositionalEncoding(d_model, tgt_seq_len)
        self.encoder = EncoderWithNewTechniques(d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = DecoderWithNewTechniques(d_model, num_layers, num_heads, d_ff, dropout)
        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        src = self.src_pos(self.src_embed(src))
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_pos(self.tgt_embed(tgt))
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

    def generate_square_subsequent_mask(self, size, device):
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def translate_batch(self, src, max_len=100, start_token_id=0, end_token_id=1):
        batch_size, device = src.size(0), src.device
        src_mask = torch.ones(batch_size, 1, 1, src.size(1), device=device).bool()
        encoder_output = self.encode(src, src_mask)
        tgt = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device=device)

            decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
            logits = self.project(decoder_output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)

            finished |= (next_token.squeeze(1) == end_token_id)
            if finished.all():
                break

        return tgt[:, 1:]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = TransformerWithNewTechniques(
        src_vocab_size=256,
        tgt_vocab_size=256,
        src_seq_len=100,
        tgt_seq_len=120,
        d_model=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        d_ff=2048
    ).to(device)

    batch_size = 2
    src = torch.randint(0, 256, (batch_size, 100)).to(device)
    tgt = torch.randint(0, 256, (batch_size, 120)).to(device)
    src_mask = torch.ones(batch_size, 1, 1, 100).to(device)
    tgt_mask = torch.tril(torch.ones(120, 120)).unsqueeze(0).unsqueeze(0).to(device)

    print("=" * 80)
    print("Input Embeddings summary:")
    summary(transformer.src_embed, input_data=src, verbose=1)

    embedded = transformer.src_embed(src)
    print("=" * 80)
    print("Positional Encoding summary:")
    summary(transformer.src_pos, input_data=embedded, verbose=1)

    encoder_input = transformer.src_pos(embedded)
    print("=" * 80)
    print("Encoder summary:")
    summary(transformer.encoder, input_data=(encoder_input, src_mask), verbose=1)

    tgt_embedded = transformer.tgt_embed(tgt)
    tgt_encoded = transformer.tgt_pos(tgt_embedded)
    encoder_output = transformer.encode(src, src_mask)
    print("=" * 80)
    print("Decoder summary:")
    summary(transformer.decoder,
            input_data=(tgt_encoded, encoder_output, src_mask, tgt_mask),
            verbose=1)

    decoder_output = transformer.decode(encoder_output, src_mask, tgt, tgt_mask)
    print("=" * 80)
    print("Projection Layer summary:")
    summary(transformer.projection_layer, input_data=decoder_output, verbose=1)

    encoder_output = transformer.encode(src, src_mask)
    decoder_output = transformer.decode(encoder_output, src_mask, tgt, tgt_mask)
    logits = transformer.project(decoder_output)

    print(f"\nEncoder output shape: {encoder_output.shape}")
    print(f"Decoder output shape: {decoder_output.shape}")
    print(f"Projection output shape: {logits.shape}")