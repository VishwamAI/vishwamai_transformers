# tests/test_models/conftest.py
import pytest
from flax import linen as nn
import jax.random as jrandom
from VishwamAI_Transformer.models.decoder import DecoderLayer

@pytest.fixture
def decoder_layer():
    config = {
        'num_heads': 2,
        'head_dim': 8,
        'dropout_rate': 0.1,
        'activation': nn.gelu
    }
    layer = DecoderLayer(**config)
    key = jrandom.PRNGKey(0)
    return layer, key