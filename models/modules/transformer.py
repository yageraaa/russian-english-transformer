import torch
import torch.nn as nn
from models.modules.encoder import Encoder
from models.modules.decoder import Decoder
from models.modules.embeddings import InputEmbeddings
from models.modules.positional_encoding import PositionalEncoding
from models.modules.linear_layer import ProjectionLayer
from models.modules.multihead_attention import MultiHeadAttention
from models.modules.feed_forward import FeedForwardLayer
from models.modules.encoder_layer import EncoderBlock
from models.modules.decoder_layer import DecoderBlock


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor,
               src_mask: torch.Tensor, tgt: torch.Tensor,
               tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor):
        return self.projection_layer(x)


def build_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048) -> Transformer:

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len)

    encoder_blocks = []
    for _ in range(num_layers):
        encoder_block = EncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        encoder_blocks.append(encoder_block)
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    decoder_blocks = []
    for _ in range(num_layers):
        decoder_block = DecoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        decoder_blocks.append(decoder_block)
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        projection_layer=projection_layer
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


if __name__ == '__main__':
    src_vocab_size = 256
    tgt_vocab_size = 256
    src_seq_len = 100
    tgt_seq_len = 120
    d_model = 512
    num_layers = 6
    num_heads = 8
    dropout = 0.1
    d_ff = 2048

    transformer = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        d_ff=d_ff
    )

    batch_size = 2
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    src_mask = torch.ones(batch_size, 1, 1, src_seq_len)
    tgt_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).unsqueeze(0).unsqueeze(0)
    encoder_output = transformer.encode(src, src_mask)
    decoder_output = transformer.decode(encoder_output, src_mask, tgt, tgt_mask)
    logits = transformer.project(decoder_output)

    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder output shape: {decoder_output.shape}")
    print(f"Projection output shape: {logits.shape}")