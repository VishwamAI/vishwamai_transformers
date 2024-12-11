import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Optional

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, query, key=None, value=None, mask=None, training=True):
        if key is None:
            key = query
        if value is None:
            value = query
        batch_size = query.shape[0]
        total_dim = self.num_heads * self.head_dim  

        # Linear projections
        query_proj = nn.Dense(features=total_dim)(query)
        key_proj = nn.Dense(features=total_dim)(key)
        value_proj = nn.Dense(features=total_dim)(value)

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        query_proj = query_proj.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_proj = key_proj.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_proj = value_proj.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim)
        attention_scores = jnp.matmul(query_proj, key_proj.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            # Ensure mask is broadcastable to attention_scores
            # mask should be of shape (batch_size, num_heads, seq_len_query, seq_len_key)
            # Add necessary dimensions if not present
            if mask.ndim == 2:
                mask = jnp.expand_dims(mask, axis=0)  # Add batch dimension
                mask = jnp.expand_dims(mask, axis=0)  # Add heads dimension
            elif mask.ndim == 3:
                mask = jnp.expand_dims(mask, axis=1)  # Add heads dimension
            attention_scores = jnp.where(mask, attention_scores, -1e9)

        attention_probs = nn.softmax(attention_scores, axis=-1)
        attention_probs = nn.Dropout(rate=self.dropout_rate)(attention_probs, deterministic=not training)

        # Apply attention to values
        output = jnp.matmul(attention_probs, value_proj)

        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, -1, total_dim)

        # Final linear projection
        return nn.Dense(features=query.shape[-1])(output)