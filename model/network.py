# model/network.py

import torch
import torch.nn as nn
from typing import List

class SpamDetectorNN(nn.Module):
    """
    Feed-Forward Neural Network for spam detection.

    ARCHITECTURE:
    Input(10010) → FC→BN→ReLU→Drop → FC→BN→ReLU→Drop → FC→BN→ReLU→Drop → FC(1)

    WHY THESE CHOICES:
    ┌─────────────────┬──────────────────────────────────────────┐
    │ Component       │ Why                                      │
    ├─────────────────┼──────────────────────────────────────────┤
    │ Linear          │ Learns weighted combination of features  │
    │ BatchNorm       │ Normalizes outputs → stable training     │
    │ ReLU            │ Adds non-linearity, fast, no vanishing   │
    │                 │ gradient problem like sigmoid has        │
    │ Dropout(0.3)    │ Randomly zeros 30% neurons → prevents   │
    │                 │ overfitting (model memorizing train data)│
    │ Sigmoid output  │ Squashes to [0,1] → spam probability    │
    └─────────────────┴──────────────────────────────────────────┘

    NOTE: We don't apply Sigmoid in forward() because
    BCEWithLogitsLoss does it internally — more numerically stable.
    We only use Sigmoid in predict_proba() during inference.
    """

    def __init__(self,
                 input_size:     int,
                 hidden_sizes:   List[int] = [512, 256, 128],
                 dropout_rate:   float     = 0.3,
                 use_batch_norm: bool      = True):

        super(SpamDetectorNN, self).__init__()

        self.input_size   = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate

        # ── Build layers dynamically ──────────────────────────────────
        layers   = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            # Linear: y = Wx + b
            layers.append(nn.Linear(prev_size, hidden_size))

            # BatchNorm: normalizes activations → faster, stable training
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # ReLU: max(0, x) — kills negatives, keeps positives
            layers.append(nn.ReLU())

            # Dropout: randomly zero out neurons during training only
            layers.append(nn.Dropout(p=dropout_rate))

            prev_size = hidden_size

        # Output layer: single neuron = spam logit
        layers.append(nn.Linear(prev_size, 1))
        # NO sigmoid here — BCEWithLogitsLoss handles it

        self.network = nn.Sequential(*layers)

        # ── He (Kaiming) initialization ───────────────────────────────
        # WHY: With ReLU, ~50% neurons are zero. He init compensates
        # so signal doesn't shrink/explode layer by layer.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Input:  (batch_size, input_size)
        Output: (batch_size, 1) — raw logits (not probabilities yet)
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns spam probability between 0 and 1.
        Use ONLY during inference, never during training.
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He init: std = sqrt(2 / fan_in)
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        lines = [
            f"\nSpamDetectorNN Architecture:",
            f"  Input size     : {self.input_size:,}",
            f"  Hidden layers  : {self.hidden_sizes}",
            f"  Dropout rate   : {self.dropout_rate}",
            f"  Total params   : {self.count_parameters():,}",
            f"\nLayer structure:"
        ]
        prev = self.input_size
        for i, h in enumerate(self.hidden_sizes):
            lines.append(f"  Layer {i+1}: Linear({prev:,} → {h}) "
                        f"→ BatchNorm → ReLU → Dropout({self.dropout_rate})")
            prev = h
        lines.append(f"  Output: Linear({prev} → 1) → [Sigmoid at inference]")
        return '\n'.join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("="*50)
    print("Testing SpamDetectorNN")
    print("="*50)

    INPUT_SIZE = 10010  # 10000 TF-IDF + 10 metadata

    model = SpamDetectorNN(
        input_size   = INPUT_SIZE,
        hidden_sizes = [512, 256, 128],
        dropout_rate = 0.3
    )

    print(model)

    # ── Test forward pass ─────────────────────────────────────────────
    print("\nTesting forward pass...")
    batch_size   = 32
    dummy_input  = torch.randn(batch_size, INPUT_SIZE)

    # Training mode (dropout active)
    model.train()
    logits = model(dummy_input)
    print(f"  Input shape  : {dummy_input.shape}")
    print(f"  Output shape : {logits.shape}")
    print(f"  Logit range  : [{logits.min().item():.3f}, "
          f"{logits.max().item():.3f}]")

    # Inference mode (dropout disabled)
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(dummy_input)
    print(f"\nTesting predict_proba (eval mode)...")
    print(f"  Probability range : [{probs.min().item():.3f}, "
          f"{probs.max().item():.3f}]")
    print(f"  All probs in [0,1]: "
          f"{'✅ YES' if (probs >= 0 ).all() and (probs <= 1).all() else '❌ NO'}")

    # ── Test loss computation ─────────────────────────────────────────
    print("\nTesting loss computation...")
    criterion = torch.nn.BCEWithLogitsLoss()
    labels    = torch.randint(0, 2, (batch_size, 1)).float()

    model.train()
    logits = model(dummy_input)
    loss   = criterion(logits, labels)
    print(f"  Loss value : {loss.item():.4f} (should be ~0.69 for random weights)")

    # ── Test backpropagation ──────────────────────────────────────────
    print("\nTesting backpropagation...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check gradients exist
    has_grads = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"  Gradients computed: {'✅ YES' if has_grads else '❌ NO'}")

    print("\n✅ Neural network working correctly!")
    print("🚀 Next step: python training/trainer.py")