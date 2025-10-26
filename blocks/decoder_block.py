import torch.nn as nn

from layers.feed_forward import FeedForwardBlock
from layers.multi_head_attention import MultiHeadAttentionBlock
from layers.residual_connection import ResidualConnection

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)]) # 3 residual connections

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x: input of the decoder
        source mask (or encoder mask): mask applied to encoder
        target mask (or decoder mask): mask applied to decoder
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x