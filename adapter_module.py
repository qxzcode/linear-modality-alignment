import torch
from torch import nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, image_embed_dim: int, text_embed_dim: int, num_output_tokens: int) -> None:
        super().__init__()
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = text_embed_dim
        self.num_output_tokens = num_output_tokens
        self.linear = nn.Linear(image_embed_dim, text_embed_dim*num_output_tokens, bias=True)

        # Initialize the weights to zeros.
        with torch.no_grad():
            self.linear.weight.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (batch_size, _) = x.shape
        x = self.linear(x)
        return x.view(batch_size, self.num_output_tokens, self.text_embed_dim)
