import torch.nn as nn

class InputEmbeddings(nn.Module):

    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
            d_model: dimension of the vector in the model
            vocab_size: number of words in the vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)