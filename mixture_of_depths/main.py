import torch
import torch.nn.functional as F
from torch import Tensor, nn
from zeta.nn import FeedForward, MultiQueryAttention


def exists(val):
    return val is not None


class LongGeminiTransformerBlock(nn.Module):
    """
    Gemini15TransformerBlock is a transformer block used in the Gemini15 model.

    Args:
        dim (int): The input dimension of the block.
        depth (int, optional): The depth of the block. Defaults to 32.
        dim_head (int, optional): The dimension of each head in the multi-head attention mechanism. Defaults to 128.
        heads (int, optional): The number of attention heads. Defaults to 24.
        use_abs_pos_emb (bool, optional): Whether to use absolute positional embeddings. Defaults to False.
        attn_flash (bool, optional): Whether to use flash attention. Defaults to True.
        attn_kv_heads (int, optional): The number of heads to use for key-value attention. Defaults to 2.
        qk_norm (bool, optional): Whether to apply layer normalization to query, key, and value. Defaults to True.
        ff_mult (int, optional): The multiplier for the hidden dimension in the feedforward network. Defaults to 4.

    Attributes:
        dim (int): The input dimension of the block.
        depth (int): The depth of the block.
        dim_head (int): The dimension of each head in the multi-head attention mechanism.
        heads (int): The number of attention heads.
        use_abs_pos_emb (bool): Whether to use absolute positional embeddings.
        attn_flash (bool): Whether to use flash attention.
        attn_kv_heads (int): The number of heads to use for key-value attention.
        qk_norm (bool): Whether to apply layer normalization to query, key, and value.
        attn (RingAttention): The attention model for the block.
        norm (nn.LayerNorm): The layer normalization module.
        ffn (FeedForward): The feedforward model for the block.

    """

    def __init__(
        self,
        dim: int,
        depth: int = 32,
        dim_head: int = 128,
        heads: int = 24,
        qk_norm: bool = True,
        ff_mult: int = 4,
        ring_seq_size: int = 512,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.heads = heads
        self.qk_norm = qk_norm
        self.ff_mult = ff_mult
        self.ring_seq_size = ring_seq_size

        # Attention model for the block
        self.attn = MultiQueryAttention(
            dim,
            heads,
        )

        # Post Attention layer normalization
        self.norm = nn.LayerNorm(dim)

        # Feedforward model for the block
        self.ffn = FeedForward(dim, dim, ff_mult, *args, **kwargs)

    def forward(self, x: Tensor, *args, **kwargs):
        """
        Forward pass of the Gemini15TransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        x = self.norm(x)

        # Attention
        x, _, _ = self.attn(x) + x

        # Feedforward
        x = self.ffn(x) + x

        return x


class MoD(nn.Module):
    def __init__(
        self,
        seq_len: int = None,
        dim: int = None,
        capacity_factor: int = None,
        transformer_block: nn.Module = None,
        vocab_size: int = None,
        aux_loss_on: bool = False,
        transformer_depth: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.capacity_factor = capacity_factor
        self.transformer_block = transformer_block
        self.aux_loss_on = aux_loss_on
        self.vocab_size = vocab_size
        self.transformer_depth = transformer_depth
        self.router = nn.Linear(dim, 1, bias=False)

        self.aux_router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
        )

        # Transformer Block
        # self.transformer = LongGeminiTransformerBlock(
        #     dim,
        #     transformer_depth,
        # )

    def forward(
        self,
        x: Tensor = None,
        mask: Tensor = None,
        freqs_cis: Tensor = None,
        *args,
        **kwargs,
    ) -> Tensor:
        b, s, d = x.shape
        device = x.device

        # Top k
        top_k = int(s * self.capacity_factor)

        # Scalar weights for each token
        router_logits = self.router(x)

        # Equation 1
        token_weights, token_index = torch.topk(
            router_logits, top_k, dim=1, sorted=False
        )

        # Selected
        selected_tokens, index = torch.sort(token_index, dim=1)

        # Select idx
        indices_expanded = selected_tokens.expand(-1, -1, self.dim)

        # Filtered topk tokens with capacity c
        filtered_x = torch.gather(
            input=x, dim=1, index=indices_expanded
        )
        print(filtered_x.shape)

        if self.transformer_block:
            x_out = self.transformer_block(x)
        else:
            x_out = filtered_x

        # Softmax router weights
        token_weights = F.softmax(token_weights, dim=1)

        # Selecting router weight by idx
        r_weights = torch.gather(token_weights, dim=1, index=index)

        # Multiply by router weights
        xw_out = r_weights * x_out

        # Out
        out = torch.scatter_add(
            input=x, dim=1, index=indices_expanded, src=xw_out
        )

        # Aux loss
        # if self.aux_loss:
        #     aux_loss = self.aux_loss(

        #     )
        if self.aux_loss_on is not False:
            aux_loss = self.aux_loss(
                out, router_logits, selected_tokens
            )
            return out, aux_loss
        return out

    def aux_loss(
        self,
        x: Tensor,
        router_logits: Tensor,
        selected_tokens: Tensor,
    ):
        b, s, d = x.shape

        router_targets = torch.zeros_like(router_logits).view(-1)

        router_targets[selected_tokens.view(-1)] = 1.0
        aux_router_logits = self.aux_router(
            x.detach().view(b * s, -1)
        )
        return F.binary_cross_entropy(
            aux_router_logits.view(-1), router_targets
        )
