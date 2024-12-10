import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
from .attention_mechanisms import MultiHeadAttention, MultiPerspectiveAttention, SparseAxialAttention

class DecoderLayer(nn.Module):
    """
    Single layer of the Decoder in the Transformer model.
    
    Args:
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        dropout_rate: Dropout rate for attention layers.
        activation: Activation function for the feed-forward network.
    """
    num_heads: int
    head_dim: int
    dropout_rate: float
    activation: Callable

    @nn.compact
    def __call__(self, x, encoder_output, training=True):
        # Self attention
        self_attention = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(x)
        
        # Add & Norm
        x = nn.LayerNorm()(x + self_attention)
        
        # Cross attention
        cross_attention = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(x, encoder_output, encoder_output)
        
        # Add & Norm
        x = nn.LayerNorm()(x + cross_attention)
        
        # Multi-Perspective Attention
        multi_perspective_attention_output = MultiPerspectiveAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(x)
        
        # Add & Norm
        x = nn.LayerNorm()(x + multi_perspective_attention_output)
        
        # Sparse/Axial Attention
        sparse_axial_attention_output = SparseAxialAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(x)
        
        # Add & Norm
        x = nn.LayerNorm()(x + sparse_axial_attention_output)
        
        # Feed Forward
        y = nn.Dense(features=4 * x.shape[-1])(x)
        y = self.activation(y)
        y = nn.Dense(features=x.shape[-1])(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
        
        # Add & Norm
        return nn.LayerNorm()(x + y)

class Decoder(nn.Module):
    """
    Decoder stack in the Transformer model, consisting of multiple Decoder layers.
    
    Args:
        num_layers: Number of decoder layers.
        num_heads: Number of attention heads in each layer.
        head_dim: Dimensionality of each attention head.
        dropout_rate: Dropout rate for attention layers.
        activation: Activation function for the feed-forward network.
    """
    num_layers: int
    num_heads: int
    head_dim: int
    dropout_rate: float
    activation: Callable

    @nn.compact
    def __call__(self, x, encoder_output, training=True):
        for _ in range(self.num_layers):
            x = DecoderLayer(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout_rate=self.dropout_rate,
                activation=self.activation
            )(x, encoder_output, training)
        return x
