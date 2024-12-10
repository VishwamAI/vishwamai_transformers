import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
from .attention_mechanisms import MultiHeadAttention

class EncoderLayer(nn.Module):
    """
    Single layer of the Encoder in the Transformer model.
    
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
    def __call__(self, x, training=True):
        # Self attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(x)
        
        # Add & Norm
        x = nn.LayerNorm()(x + attention_output)
        
        # Feed Forward
        y = nn.Dense(features=4 * x.shape[-1])(x)
        y = self.activation(y)
        y = nn.Dense(features=x.shape[-1])(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
        
        # Add & Norm
        return nn.LayerNorm()(x + y)

class Encoder(nn.Module):
    """
    Encoder stack in the Transformer model, consisting of multiple Encoder layers.
    
    Args:
        num_layers: Number of encoder layers.
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
    def __call__(self, x, training=True):
        for _ in range(self.num_layers):
            x = EncoderLayer(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout_rate=self.dropout_rate,
                activation=self.activation
            )(x, training)
        return x