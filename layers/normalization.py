import torch
import torch.nn as nn

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps # epsilon for numerical stability and to avoid divide by zero error
        self.alpha = nn.Parameter(torch.ones(features)) # alpha (multiplied) is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias (added) is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        #calculate standard dev of the last dimenison. i.e everything after the batch.
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias