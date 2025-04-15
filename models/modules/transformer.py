import torch
import torch.nn as nn
from models.modules.encoder import Encoder
from models.modules.decoder import Decoder
from models.modules.embeddings import InputEmbeddings
from models.modules.positional_encoding import PositionalEncoding
from models.modules.linear_layer import ProjectionLayer
from torchinfo import summary


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len,
                 d_model=512, num_layers=6, num_heads=8, dropout=0.1, d_ff=2048):
        super().__init__()
        self.src_embed = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        self.src_pos = PositionalEncoding(d_model, src_seq_len)
        self.tgt_pos = PositionalEncoding(d_model, tgt_seq_len)

        self.encoder = Encoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        self.decoder = Decoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = Transformer(
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