import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention mechanism.
    """
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask: Optional[jnp.ndarray] = None, training: bool = True):
        batch_size = x.shape[0]
        total_dim = self.num_heads * self.head_dim  # Compute total_dim directly
        
        # Linear projections
        query = nn.Dense(features=total_dim)(x)
        key = nn.Dense(features=total_dim)(x)
        value = nn.Dense(features=total_dim)(x)
        
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        query = query.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim)
        attention = jnp.matmul(query, key.transpose(0, 1, 3, 2)) / scale
        
        if mask is not None:
            attention = jnp.where(mask, attention, -1e9)
        
        attention = nn.softmax(attention)
        attention = nn.Dropout(rate=self.dropout_rate)(attention, deterministic=not training)
        
        # Apply attention to values
        output = jnp.matmul(attention, value)
        
        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, -1, total_dim)  # Use total_dim directly
        
        # Final linear projection
        return nn.Dense(features=x.shape[-1])(output)
class RelativePositionalEncodingAttention(nn.Module):
    """
    Multi-Head Attention with Relative Positional Encodings.
    """
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1
    max_relative_position: int = 10
    
    @nn.compact
    def __call__(self, query, key, value, mask=None, training=False):
        embed_dim = self.num_heads * self.head_dim
        
        # Project query, key, and value
        q_proj = nn.Dense(embed_dim, use_bias=False)(query)
        k_proj = nn.Dense(embed_dim, use_bias=False)(key)
        v_proj = nn.Dense(embed_dim, use_bias=False)(value)
        
        # Split into multiple heads
        batch_size, seq_length, _ = q_proj.shape
        q = jnp.reshape(q_proj, (batch_size, seq_length, self.num_heads, self.head_dim))
        k = jnp.reshape(k_proj, (batch_size, seq_length, self.num_heads, self.head_dim))
        v = jnp.reshape(v_proj, (batch_size, seq_length, self.num_heads, self.head_dim))
        
        q = q.transpose((0, 2, 1, 3))  # (batch_size, num_heads, seq_length, head_dim)
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))
        
        # Relative positional embeddings
        relative_positions = jnp.arange(seq_length)[:, None] - jnp.arange(seq_length)[None, :]
        relative_positions = jnp.clip(relative_positions, -self.max_relative_position, self.max_relative_position)
        relative_positions += self.max_relative_position
        pos_embeddings = self.param('pos_embeddings', nn.initializers.normal(), 
                                    (2 * self.max_relative_position + 1, self.num_heads))
        pos_embeddings = pos_embeddings[relative_positions]
        
        # Add positional embeddings to attention scores
        scores = jnp.matmul(q, k.transpose((0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        scores = scores + pos_embeddings[:, :, None, :]
        
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)
            
        attn_probs = nn.softmax(scores, axis=-1)
        attn_probs = nn.Dropout(rate=self.dropout_rate)(attn_probs, deterministic=not training)
        
        # Multiply with value matrices
        attn_output = jnp.matmul(attn_probs, v)
        
        # Transpose and concatenate the heads
        attn_output = attn_output.transpose((0, 2, 1, 3))
        attn_output = jnp.reshape(attn_output, (batch_size, seq_length, embed_dim))
        
        # Final projection
        return nn.Dense(query.shape[-1], use_bias=False)(attn_output)

class LocalAttention(nn.Module):
    """
    Local Attention mechanism with a fixed window size.
    """
    window_size: int
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1
    causal: bool = False
    
    @nn.compact
    def __call__(self, query, key, value, mask=None, training=False):
        embed_dim = self.num_heads * self.head_dim
        
        # Project query, key, and value
        q_proj = nn.Dense(embed_dim, use_bias=False)(query)
        k_proj = nn.Dense(embed_dim, use_bias=False)(key)
        v_proj = nn.Dense(embed_dim, use_bias=False)(value)
        
        # Split into multiple heads
        batch_size, seq_length, _ = q_proj.shape
        q = jnp.reshape(q_proj, (batch_size, seq_length, self.num_heads, self.head_dim))
        k = jnp.reshape(k_proj, (batch_size, seq_length, self.num_heads, self.head_dim))
        v = jnp.reshape(v_proj, (batch_size, seq_length, self.num_heads, self.head_dim))
        
        q = q.transpose((0, 2, 1, 3))  # (batch_size, num_heads, seq_length, head_dim)
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))
        
        # Create local mask
        local_mask = jnp.ones((seq_length, seq_length), dtype=bool)
        for i in range(seq_length):
            start = max(0, i - self.window_size)
            end = min(seq_length, i + self.window_size + 1)
            local_mask = local_mask.at[i, start:end].set(False)
        
        if self.causal:
            causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
            local_mask = local_mask | causal_mask
        
        # Combine with existing mask
        if mask is not None:
            local_mask = local_mask & mask
        
        # Scaled dot-product attention with local mask
        scores = jnp.matmul(q, k.transpose((0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        scores = jnp.where(local_mask, -jnp.inf, scores)
        
        attn_probs = nn.softmax(scores, axis=-1)
        attn_probs = nn.Dropout(rate=self.dropout_rate)(attn_probs, deterministic=not training)
        
        # Multiply with value matrices
        attn_output = jnp.matmul(attn_probs, v)
        
        # Transpose and concatenate the heads
        attn_output = attn_output.transpose((0, 2, 1, 3))
        attn_output = jnp.reshape(attn_output, (batch_size, seq_length, embed_dim))
        
        # Final projection
        return nn.Dense(query.shape[-1], use_bias=False)(attn_output)