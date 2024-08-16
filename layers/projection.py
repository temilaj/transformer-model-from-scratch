import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    """
    Linear layer projecting the embedding into the vocabulary. i.e to convert embedding back into position in the vocabulary
    """

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        # return self.proj(x)
        return torch.log_softmax(self.proj(x), dim=-1)