import jax
import jax.numpy as jnp
from flax import linen as nn
from src.models.decoder import DecoderLayer, Decoder
import pytest

# Fixture for DecoderLayer
@pytest.fixture
def decoder_layer():
    config = {
        'num_heads': 2,
        'head_dim': 8,
        'dropout_rate': 0.1,
        'activation': nn.gelu
    }
    layer = DecoderLayer(**config)
    key = jax.random.PRNGKey(0)  # Proper JAX PRNGKey
    return layer, key

def test_decoder_layer_output_shape(decoder_layer):
    layer, key = decoder_layer
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    enc_output = jnp.ones((batch_size, seq_len, embed_dim))
    variables = layer.init(key, x, enc_output)  # Use the key directly
    output = layer.apply(variables, x, enc_output)
    assert output.shape == (batch_size, seq_len, embed_dim)

# Test for Decoder stack
def test_decoder_output_shape(decoder_layer):
    config = {
        'num_layers': 2,
        'num_heads': 2,
        'head_dim': 8,
        'dropout_rate': 0.1,
        'activation': nn.gelu
    }
    decoder = Decoder(**config)
    key = jax.random.PRNGKey(0)  # Proper JAX PRNGKey
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    enc_output = jnp.ones((batch_size, seq_len, embed_dim))
    variables = decoder.init(key, x, enc_output)  # Use the key directly
    output = decoder.apply(variables, x, enc_output)
    assert output.shape == (batch_size, seq_len, embed_dim)