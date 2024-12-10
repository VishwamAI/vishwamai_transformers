import jax
import jax.numpy as jnp
from flax import linen as nn
from VishwamAI_Transformer.models.attention_mechanisms import MultiHeadAttention
from VishwamAI_Transformer.models.decoder import DecoderLayer, Decoder
from VishwamAI_Transformer.models.encoder import EncoderLayer, Encoder

def test_multi_head_attention_output_shape():
    attention = MultiHeadAttention(num_heads=2, head_dim=8, dropout_rate=0.1)
    key = jax.random.PRNGKey(0)
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    
    variables = attention.init(key, x)
    output = attention.apply(variables, x, rngs={'dropout': key})
    assert output.shape == (batch_size, seq_len, embed_dim)

def test_decoder_layer_output_shape():
    layer = DecoderLayer(num_heads=2, head_dim=8, dropout_rate=0.1, activation=nn.gelu)
    key = jax.random.PRNGKey(0)
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

def test_encoder_layer_output_shape():
    layer = EncoderLayer(num_heads=2, head_dim=8, dropout_rate=0.1, activation=nn.gelu)
    key = jax.random.PRNGKey(0)
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    mask = jnp.ones((batch_size, seq_len, seq_len))  # Example mask
    
    variables = layer.init(key, x)
    output = layer.apply(variables, x, mask=mask, rngs={'dropout': key})
    assert output.shape == (batch_size, seq_len, embed_dim)

def test_encoder_output_shape():
    config = {
        'num_layers': 2,
        'num_heads': 2,
        'head_dim': 8,
        'dropout_rate': 0.1,
        'activation': nn.gelu
    }
    encoder = Encoder(**config)
    key = jax.random.PRNGKey(0)
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    mask = jnp.ones((batch_size, seq_len, seq_len))  # Example mask
    
    variables = encoder.init(key, x)
    output = encoder.apply(variables, x, mask=mask, rngs={'dropout': key})
    assert output.shape == (batch_size, seq_len, embed_dim)

def test_multi_perspective_attention_in_encoder_layer():
    layer = EncoderLayer(num_heads=2, head_dim=8, dropout_rate=0.1, activation=nn.gelu)
    key = jax.random.PRNGKey(0)
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    mask = jnp.ones((batch_size, seq_len, seq_len))  # Example mask
    
    variables = layer.init(key, x)
    output = layer.apply(variables, x, mask=mask, rngs={'dropout': key})
    assert output.shape == (batch_size, seq_len, embed_dim)

def test_sparse_axial_attention_in_encoder_layer():
    layer = EncoderLayer(num_heads=2, head_dim=8, dropout_rate=0.1, activation=nn.gelu)
    key = jax.random.PRNGKey(0)
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    mask = jnp.ones((batch_size, seq_len, seq_len))  # Example mask
    
    variables = layer.init(key, x)
    output = layer.apply(variables, x, mask=mask, rngs={'dropout': key})
    assert output.shape == (batch_size, seq_len, embed_dim)
