[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

## Mixture of Depths Scaling
Implementation of the paper: "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models". From the paper: "These models match baseline performance for equivalent FLOPS and wall-clock times to train, but require a fraction of the FLOPs per forward pass, and can be upwards of 50% faster to step during post-training sampling."


## install 
`pip3 install mixture-of-depths`

## usage
```python
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
```

# License
MIT
