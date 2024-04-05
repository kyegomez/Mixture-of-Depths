import torch
from torch import nn, Tensor


class TokenRouterPredictor(nn.Module):
    """A router to emit scalar weights for each token."""

    def __init__(self, embed_dim):
        super(TokenRouterPredictor, self).__init__()
        # Simple linear layer to generate a scalar weight for each token
        self.weight_predictor = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        weights = self.weight_predictor(x)  # [batch_size, seq_len, 1]
        return weights


class MoDRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        token_limit: int,
        block: nn.Module,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        # Limit of percentage of tokens to be attended to
        self.token_limit = token_limit
        self.block = block

        # Token router Predictor
        self.token_predictor = TokenRouterPredictor(dim)

        # Get the number of tokens to be attended to
        num_tokens = int(self.token_limit * self.num_tokens)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_len, embed_dim]
        # Get the weights for each token
        weights = self.token_predictor(x)

        # Apply softmax to get the routing probabilities
        weights = torch.softmax(weights, dim=-1)

        # Get the top-k tokens
        _, indices = torch.topk(weights, self.num_tokens, dim=1)

        # Get the top-k token embeddings
        topk_tokens = torch.gather(
            x, 1, indices.expand(-1, self.num_tokens, -1)
        )

        # Apply the block to the top-k tokens
        out = self.block(topk_tokens)

        return out
