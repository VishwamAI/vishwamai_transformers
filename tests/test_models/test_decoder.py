import jax
import jax.numpy as jnp
from flax import linen as nn
from VishwamAI_Transformer.models.decoder import DecoderLayer, Decoder
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
    key = jax.random.PRNGKey(0)
    return layer, key

def test_decoder_layer_output_shape(decoder_layer):
    layer, key = decoder_layer
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    enc_output = jnp.ones((batch_size, seq_len, embed_dim))
    variables = layer.init(key, x, enc_output)
    output = layer.apply(variables, x, enc_output, rngs={'dropout': key})
    assert output.shape == (batch_size, seq_len, embed_dim)

def test_decoder_output_shape():
    config = {
        'num_layers': 2,
        'num_heads': 2,
        'head_dim': 8,
        'dropout_rate': 0.1,
        'activation': nn.gelu
    }
    decoder = Decoder(**config)
    key = jax.random.PRNGKey(0)
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    enc_output = jnp.ones((batch_size, seq_len, embed_dim))
    
    variables = decoder.init(key, x, enc_output)
    output = decoder.apply(variables, x, enc_output, rngs={'dropout': key})
    assert output.shape == (batch_size, seq_len, embed_dim)