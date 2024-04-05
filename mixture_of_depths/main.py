# import torch
# from torch import nn, Tensor
# from einops import reduce

# class TokenRouterPredictor(nn.Module):
#     """A router to emit scalar weights for each token."""

#     def __init__(self, dim):
#         super(TokenRouterPredictor, self).__init__()
#         # Simple linear layer to generate a scalar weight for each token
#         self.weight_predictor = nn.Linear(dim, 1)

#     def forward(self, x):
#         # x: [batch_size, seq_len, dim]
#         weights = self.weight_predictor(x)  # [batch_size, seq_len, 1]
#         return weights


# class MoDRouter(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_tokens: int,
#         token_limit: int,
#         block: nn.Module,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.num_tokens = num_tokens
#         # Limit of percentage of tokens to be attended to
#         self.token_limit = token_limit
#         # self.block = block

#         # Token router Predictor
#         self.token_predictor = TokenRouterPredictor(dim)

#         # Get the number of tokens to be attended to
#         num_tokens = int(self.token_limit * self.num_tokens)

#     def forward(self, x: Tensor) -> Tensor:
#         # x: [batch_size, seq_len, dim]
#         # Get the weights for each token
#         weights = self.token_predictor(x)
#         print(weights)
#         weights = reduce(weights, "b s d -> d", "min")

#         # # Apply softmax to get the routing probabilities
#         # weights = torch.softmax(weights, dim=-1)

#         # # Get the top-k tokens
#         # _, indices = torch.topk(weights, self.num_tokens, dim=1)

#         # # Get the top-k token embeddings
#         # topk_tokens = torch.gather(
#         #     x, 1, indices.expand(-1, self.num_tokens, -1)
#         # )

#         # Apply the block to the top-k tokens
#         # out = self.block(weights)

#         return weights


# # Random tensor of shape [batch_size, seq_len, dim]
# x = torch.randn(2, 10, 64)

# # Create the MoD Router
# mod_router = MoDRouter(64, 10, 0.5, block=None) #nn.Linear(64, 64))

# # Get the output
# out = mod_router(x)
# print(out.shape)  # torch.Size([2, 10, 64])


import torch
from torch import nn, Tensor


class TokenRouter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight_predictor = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weights = self.weight_predictor(x).squeeze(
            -1
        )  # [batch_size, seq_len]
        return weights


class MoD(nn.Module):
    def __init__(self, capacity, block):
        super().__init__()
        self.router = TokenRouter(block.dim)
        self.block = block
        self.capacity = capacity

    def forward(self, x: Tensor):
        b, s, d = x.shape
        weights = self.router(x)

        # Compute B-th percentil for router weightsto determine the capacity threshold
        k = int(self.capacity * s)
        top_k_values, _ = torch.topk(weights, k, dim=1, sorted=True)
        threshold = top_k_values[:, :, -1]

        # Determine which tokens exceed the threshold
        selected_mask = weights > threshold.unsqueeze(-1)

        # Process onlys elected tokens through the block
        processed_tokens = torch.zeros_like(x)
        for i in range(b):
            # Process tokens for each block
            selected_tokens = x[i][selected_mask[i]]
            if selected_tokens.size(0) > 0:
                processed_tokens[i][selected_mask[i]] = self.block(
                    selected_tokens.unsqueeze(0)
                ).squeeze(0)

        # Combine processed tokens with unprocessed ones
        output = processed_tokens + (
            x * (~selected_mask).unsqueeze(-1).to(x.dtype)
        )
        return output
