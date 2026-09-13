"""Learnable positional encoding for matrix-graph nodes."""

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for spatial coordinates

    Maps spatial coordinates to a fixed-dimensional feature space using
    a multi-layer perceptron with layer normalization.

    Args:
        input_dim: Dimension of input coordinates (2 for 2D, 3 for 3D, etc.)
        output_dim: Dimension of output positional encoding (default: 16)
        hidden_dim: Dimension of hidden layers (default: 32)
        num_layers: Number of hidden layers (default: 2)
        activation: Activation function (default: 'relu')

    Example:
        >>> pe = LearnablePositionalEncoding(input_dim=2, output_dim=16)
        >>> coords = torch.randn(100, 2)  # 100 nodes with 2D coordinates
        >>> pos_enc = pe(coords)  # (100, 16)
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 2,
        activation: str = 'relu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Select activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build encoder layers
        layers = []

        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn()
        ])

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn()
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode spatial coordinates to positional features

        Args:
            coords: Spatial coordinates, shape (N, input_dim)

        Returns:
            pos_enc: Positional encoding, shape (N, output_dim)
        """
        return self.encoder(coords)

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


