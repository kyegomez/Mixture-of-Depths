[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

## Mixture of Depths Ölçeklendirmesi
"Mixture-of-Depths: Dynamically allocating compute in transformer-based language models" makalesinin uygulamasıdır. Makaleden: "Bu modeller, eğitim için eşdeğer FLOP ve gerçek (wall-clock) sürelerde temel performansla eşleşir; ancak ileri geçiş başına düşen FLOP miktarının yalnızca küçük bir kısmına ihtiyaç duyarlar ve eğitim sonrası örnekleme adımlarında %50'ye kadar daha hızlı olabilirler."

## İndir
`pip3 install mixture-of-depths`

## Kullanım
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

# Lisans
MIT
