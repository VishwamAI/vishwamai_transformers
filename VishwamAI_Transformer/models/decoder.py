import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
from .attention_mechanisms import MultiHeadAttention

class DecoderLayer(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float
    activation: Callable

    @nn.compact
    def __call__(self, x, encoder_output, training=True):
        # Generate causal mask
        seq_len = x.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        causal_mask = jnp.expand_dims(causal_mask, axis=0)  # Add batch dimension
        causal_mask = jnp.expand_dims(causal_mask, axis=1)  # Add heads dimension

        # Self attention
        self_attention = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(query=x, key=x, value=x, mask=causal_mask, training=training)

        # Add & Norm
        x = nn.LayerNorm()(x + self_attention)

        # Cross attention
        cross_attention = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(query=x, key=encoder_output, value=encoder_output, mask=None, training=training)

        # Add & Norm
        x = nn.LayerNorm()(x + cross_attention)

        # Feed Forward
        y = nn.Dense(features=4 * x.shape[-1])(x)
        y = self.activation(y)
        y = nn.Dense(features=x.shape[-1])(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)

        # Add & Norm
        return nn.LayerNorm()(x + y)

class Decoder(nn.Module):
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