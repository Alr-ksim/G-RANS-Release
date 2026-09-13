"""Residual-aware multi-scale Poly-GAT basis generator."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional

from .positional_encoding import LearnablePositionalEncoding


class PolyGATWithPE(nn.Module):
    """Poly-GAT with Learnable Positional Encoding

    Architecture:
        1. Learnable positional encoding: coords -> pos_enc
        2. Multi-layer GATv2Conv backbone: simulates A^k operations
        3. Per-layer projection heads: extract basis from each scale
        4. Dense concatenation: combine all scales
        5. Dense candidate block returned to the solver/trainer

    Args:
        coord_dim: Dimension of input coordinates (2 for 2D, 3 for 3D)
        pos_enc_dim: Dimension of positional encoding (default: 16)
        hidden_channels: Hidden layer dimension (default: 48)
        num_layers: Number of GNN layers (default: 4)
        basis_per_layer: Number of basis vectors per layer (default: 5)
        heads: Number of attention heads (default: 3)
        dropout: Dropout probability (default: 0.08)
        use_residual_norm: Whether to normalize input residual by graph
        use_qr: Compatibility option for standalone use. The G-RANS factory
            disables it because basis accumulation performs the only QR needed.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        pos_enc_dim: int = 16,
        hidden_channels: int = 48,
        num_layers: int = 4,
        basis_per_layer: int = 5,
        heads: int = 3,
        dropout: float = 0.08,
        use_residual_norm: bool = True,
        use_qr: bool = False,
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.pos_enc_dim = pos_enc_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.basis_per_layer = basis_per_layer
        self.heads = heads
        self.dropout = dropout
        self.use_residual_norm = use_residual_norm
        self.use_qr = use_qr

        # Total basis size
        self.basis_size = num_layers * basis_per_layer

        # Input channels: residual (1) + positional encoding (pos_enc_dim)
        self.in_channels = 1 + pos_enc_dim

        # 1. Positional encoding module
        self.pos_encoder = LearnablePositionalEncoding(
            input_dim=coord_dim,
            output_dim=pos_enc_dim,
            hidden_dim=32,
            num_layers=2,
            activation='relu'
        )

        # 2. Backbone: Multi-layer GAT (simulates power iteration A^k)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer: in_channels -> hidden_channels
        self.convs.append(
            GATv2Conv(
                self.in_channels,
                hidden_channels,
                heads=heads,
                edge_dim=1,
                dropout=dropout,
                concat=False  # Average heads for consistent dimensions
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Subsequent layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    hidden_channels,
                    hidden_channels,
                    heads=heads,
                    edge_dim=1,
                    dropout=dropout,
                    concat=False
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # 3. Projection heads: Extract basis vectors from each layer
        self.projection_heads = nn.ModuleList()
        for _ in range(num_layers):
            self.projection_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels // 2, basis_per_layer)
                )
            )

    def forward(
        self,
        residual: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with positional encoding and multi-scale basis generation

        Args:
            residual: Residual vector, shape (N, 1) or (N,)
            coords: Spatial coordinates, shape (N, coord_dim)
            edge_index: Edge indices, shape (2, nnz)
            edge_attr: Edge attributes (matrix values), shape (nnz, 1)
            batch: Batch assignment, shape (N,) or None for single graph

        Returns:
            Candidate basis block, shape (N, num_layers * basis_per_layer).
            It is orthogonal only when the compatibility option ``use_qr`` is set.
        """
        # Handle batch index
        if batch is None:
            batch = torch.zeros(residual.size(0), dtype=torch.long, device=residual.device)

        # Ensure residual is 2D
        if residual.dim() == 1:
            residual = residual.unsqueeze(1)  # (N,) -> (N, 1)

        # 1. Encode spatial coordinates to positional features
        pos_enc = self.pos_encoder(coords)  # (N, pos_enc_dim)

        # 2. Concatenate residual and positional encoding
        x = torch.cat([residual, pos_enc], dim=1)  # (N, 1 + pos_enc_dim)

        # 3. Per-graph residual normalization
        if self.use_residual_norm:
            x = self._normalize_residuals(x, batch)

        # 4. Multi-scale basis extraction
        basis_parts = []
        h = x

        # Forward through each layer and extract basis
        for i, (conv, bn, proj) in enumerate(
            zip(self.convs, self.batch_norms, self.projection_heads)
        ):
            # GNN propagation (simulates A^(i+1) operation)
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = bn(h)
            h = F.elu(h)

            # Extract basis vectors from this scale
            v_k = proj(h)  # (N, basis_per_layer)
            basis_parts.append(v_k)

        # 5. Concatenate all scales into raw basis
        V_raw = torch.cat(basis_parts, dim=1)  # (N, num_layers * basis_per_layer)

        # 6. Optional compatibility QR for standalone model use
        if self.use_qr:
            Q = self._batch_qr_decomposition(V_raw, batch)
            return Q
        else:
            return V_raw

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def _normalize_residuals(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Normalize residual channel per graph to unit L2 norm

        Args:
            x: Node features (N, in_channels), first channel is residual
            batch: Batch assignment (N,)

        Returns:
            x_normalized: Normalized features
        """
        x_norm = x.clone()
        r = x[:, 0]  # Residual channel

        # Compute L2 norm per graph and normalize to unit norm
        num_graphs = batch.max().item() + 1
        for i in range(num_graphs):
            mask = (batch == i)
            r_graph = r[mask]
            r_norm = torch.norm(r_graph)
            x_norm[mask, 0] = r_graph / (r_norm + 1e-10)

        return x_norm

    def _batch_qr_decomposition(
        self,
        V_raw: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Batch-aware QR decomposition for orthogonalization

        Args:
            V_raw: Raw basis vectors (N, basis_size)
            batch: Batch assignment (N,)

        Returns:
            Q: Orthogonalized basis vectors (N, basis_size)
        """
        Q = torch.zeros_like(V_raw)
        num_graphs = batch.max().item() + 1

        for i in range(num_graphs):
            mask = (batch == i)
            V_graph = V_raw[mask]  # (N_i, basis_size)

            # QR decomposition in FP32 for numerical stability
            V_fp32 = V_graph.float()
            Q_graph, _ = torch.linalg.qr(V_fp32, mode='reduced')
            Q_graph = Q_graph.to(V_raw.dtype)

            Q[mask] = Q_graph

        return Q


# Public method name while retaining the historical class name in checkpoints.
ResidualBasisGenerator = PolyGATWithPE
