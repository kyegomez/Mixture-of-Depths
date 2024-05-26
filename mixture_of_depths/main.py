import torch 
from torch import nn, Tensor
import torch.nn.functional as F

class MoD(nn.Module):
    def __init__(
        self,
        seq_len: int = None,
        dim: int = None,
        capacity_factor: int = None,
        transformer_block: nn.Module = None,
        aux_loss: bool = True,
        *args,
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.capacity_factor = capacity_factor
        self.transformer_block = transformer_block
        self.aux_loss = aux_loss
        self.router = nn.Linear(dim, 1, bias=False)
        
        self.aux_router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(self, x: Tensor, mask: Tensor, freqs_cis: Tensor, *args, **kwargs) -> Tensor:
        b, s, d = x.shape
        device = x.device
        
        # Top k
        top_k = int(s * self.capacity_factor)
        
        # Scalar weights for each token
        router_logits = self.router(x)
        
        # Equation 1
        token_weights, token_index = torch.topk(
            router_logits,
            top_k,
            dim=1,
            sorted=False
        )
        
        # Selected
        selected_tokens, index = torch.sort(token_index, dim=1)
        
        # Select idx
        indices_expanded = selected_tokens.expand(-1, -1, self.dim)
        
        # Filtered topk tokens with capacity c
        filtered_x = torch.gather(
            input = x,
            dim = 1,
            index = indices_expanded
        )
        
        # X
        x_out, _ = self.transformer_block(x, mask, freqs_cis)
        
        # Softmax router weights
        token_weights = F.softmax(token_weights, dim=1)
        
        # Selecting router weight by idx
        r_weights = torch.gather(token_weights, dim=1, index=index)
        
        # Multiply by router weights
        xw_out = r_weights * x_out
        
        # Out
        out = torch.scatter_add(
            input = x, 
            dim = 1,
            index = indices_expanded,
            src = xw_out
        )

        # Aux loss
        # if self.aux_loss:
        #     aux_loss = self.aux_loss(
                
        #     )
        if self.aux_loss:
            aux_loss = self.aux_loss(
                out,
                router_logits,
                selected_tokens
            )
            return out, aux_loss
        return out, _
        
    def aux_loss(
        self,
        x: Tensor,
        router_logits: Tensor,
        selected_tokens: Tensor
    ):
        b, s, d = x.shape
        
        router_targets = torch.zeros_like(router_logits).view(-1)
        
        router_targets[selected_tokens.view(-1)] = 1.0
        aux_router_logits = self.aux_router(
            x.detach().view(b * s, -1)
        )
        return F.binary_cross_entropy(aux_router_logits.view(-1), router_targets)
    
    
    