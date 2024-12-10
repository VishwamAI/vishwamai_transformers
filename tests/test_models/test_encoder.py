import jax
import jax.numpy as jnp
from flax import linen as nn
from src.models.encoder import EncoderLayer, Encoder
import pytest

# Fixture for EncoderLayer
@pytest.fixture
def encoder_layer():
    config = {
        'num_heads': 2,
        'head_dim': 8,
        'dropout_rate': 0.1,
        'activation': nn.gelu
    }
    layer = EncoderLayer(**config)
    key = jax.random.PRNGKey(0)  # Proper JAX PRNGKey
    return layer, key

def test_encoder_layer_output_shape(encoder_layer):
    layer, key = encoder_layer
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    variables = layer.init(key, x)  # Use the key directly
    output = layer.apply(variables, x)
    assert output.shape == (batch_size, seq_len, embed_dim)

# Test for Encoder stack
def test_encoder_output_shape(encoder_layer):
    config = {
        'num_layers': 2,
        'num_heads': 2,
        'head_dim': 8,
        'dropout_rate': 0.1,
        'activation': nn.gelu
    }
    encoder = Encoder(**config)
    key = jax.random.PRNGKey(0)  # Proper JAX PRNGKey
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    variables = encoder.init(key, x)  # Use the key directly
    output = encoder.apply(variables, x)
    assert output.shape == (batch_size, seq_len, embed_dim)