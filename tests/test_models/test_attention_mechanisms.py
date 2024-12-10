import jax
import jax.numpy as jnp
from flax import linen as nn
from VishwamAI_Transformer.models.attention_mechanisms import MultiHeadAttention
import pytest

@pytest.fixture
def multi_head_attention():
    attention = MultiHeadAttention(
        num_heads=2,
        head_dim=8,
        dropout_rate=0.1
    )
    key = jax.random.PRNGKey(0)  # Proper JAX PRNGKey
    return attention, key

def test_multi_head_attention_output_shape(multi_head_attention):
    attention, key = multi_head_attention
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    
    # Split the key for initialization
    init_key = key
    
    variables = attention.init(init_key, x)
    output = attention.apply(variables, x, rngs={'dropout': key})
    assert output.shape == (batch_size, seq_len, embed_dim)