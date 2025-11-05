# test_full_model.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from models.hypformer import Classifier

print("=" * 60)
print("TESTING COMPLETE HYPERBOLIC CLASSIFIER")
print("=" * 60)

# Config
vocab_size = 1000
pad_id = 0
embed_dim = 128
num_classes = 5
B, N = 2, 10

# Create model
model = Classifier(
    vocab_size=vocab_size,
    pad_id=pad_id,
    embed_dim=embed_dim,
    num_classes=num_classes,
    att_type='full',
    decoder_type='linear',
)

print(f"✓ Model created")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test input
token_ids = torch.randint(0, vocab_size, (B, N))
token_ids[:, -2:] = pad_id  # Add some padding
mask = token_ids != pad_id

print(f"\nInput shape: {token_ids.shape}")
print(f"Mask: {mask}")

# Forward pass
logits = model(token_ids, mask)

print(f"\n✓ Forward pass successful!")
print(f"  Output shape: {logits.shape}")
print(f"  Expected: [{B}, {num_classes}]")
print(f"  Has NaN: {torch.isnan(logits).any()}")
print(f"  Sample logits: {logits[0]}")

print("\n" + "=" * 60)
print("✅ FULL MODEL TEST COMPLETE")
print("=" * 60)