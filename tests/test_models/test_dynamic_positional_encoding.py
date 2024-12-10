import jax
import jax.numpy as jnp
from flax import linen as nn
from VishwamAI_Transformer.models.positional_encoding import DynamicPositionalEncoding
import pytest

@pytest.fixture
def dynamic_positional_encoding():
    d_model = 512
    pe = DynamicPositionalEncoding(d_model=d_model)
    key = jax.random.PRNGKey(0)
    return pe, key

def test_dynamic_positional_encoding_output_shape(dynamic_positional_encoding):
    pe, key = dynamic_positional_encoding
    batch_size, seq_len, d_model = 4, 10, 512
    x = jnp.ones((batch_size, seq_len, d_model))
    positions = jnp.arange(seq_len).reshape(1, -1).repeat(batch_size, axis=0)
    
    variables = pe.init(key, x, positions)
    output = pe.apply(variables, x, positions)
    assert output.shape == (batch_size, seq_len, d_model)

def test_dynamic_positional_encoding_values(dynamic_positional_encoding):
    pe, key = dynamic_positional_encoding
    batch_size, seq_len, d_model = 4, 10, 512
    x = jnp.ones((batch_size, seq_len, d_model))
    positions = jnp.arange(seq_len).reshape(1, -1).repeat(batch_size, axis=0)
    
    variables = pe.init(key, x, positions)
    output = pe.apply(variables, x, positions)
    
    # Check if the positional encodings are added correctly
    div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32) * (-jnp.log(10000.0) / d_model))
    expected_pe = jnp.zeros((batch_size, seq_len, d_model))
    expected_pe = expected_pe.at[:, :, 0::2].set(jnp.sin(positions * div_term))
    expected_pe = expected_pe.at[:, :, 1::2].set(jnp.cos(positions * div_term))
    
    assert jnp.allclose(output - x, expected_pe, atol=1e-6)
