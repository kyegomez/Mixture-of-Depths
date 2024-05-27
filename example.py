import torch
from mixture_of_depths.main import MoD

x = torch.randn(1, 1000, 512)
# mask = torch.ones(1)

# Model
model = MoD(
    seq_len=1000,
    dim=512,
    capacity_factor=0.12,
    vocab_size=10000,
    transformer_depth=8,
)

# Model
out = model(x)
print(out)
