import jax
import jax.numpy as jnp
from flax import linen as nn
from VishwamAI_Transformer.models.encoder import EncoderLayer, Encoder

def test_encoder_layer_output_shape():
    layer = EncoderLayer(num_heads=2, head_dim=8, dropout_rate=0.1, activation=nn.gelu)
    key = jax.random.PRNGKey(0)
    batch_size, seq_len, embed_dim = 4, 10, 16
    x = jnp.ones((batch_size, seq_len, embed_dim))
    
    variables = layer.init(key, x)
    output = layer.apply(variables, x, rngs={'dropout': key})
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
    
    variables = encoder.init(key, x)
    output = encoder.apply(variables, x, rngs={'dropout': key})
    assert output.shape == (batch_size, seq_len, embed_dim)