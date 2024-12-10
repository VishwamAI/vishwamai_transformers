import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable

# Import custom modules
from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding

class TransformerBase(nn.Module):
    """
    Base Transformer model architecture.
    
    Args:
        num_layers: Number of encoder and decoder layers.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        dropout_rate: Dropout rate.
        activation: Activation function to use.
    """
    num_layers: int
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1
    activation: Callable = nn.gelu

    def setup(self):
        self.encoder = Encoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        )
        
        self.decoder = Decoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        )

    def __call__(self, src, tgt, mask=None, training=True):
        encoder_output = self.encoder(src, mask=mask, training=training)
        decoder_output = self.decoder(tgt, encoder_output, mask=mask, training=training)
        return decoder_output

def test_transformer_base():
    """
    Test function to verify the TransformerBase model.
    """
    # Define a simple config
    config = {
        'src_vocab_size': 10000,
        'tgt_vocab_size': 10000,
        'max_len': 100,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'num_heads': 8,
        'head_dim': 64,
        'd_model': 512,
        'dropout_rate': 0.1,
        'activation': nn.gelu
    }
    config = type('Config', (object,), config)
    
    # Create model instance
    transformer = TransformerBase(config=config)
    
    # Create dummy inputs
    key = jax.random.PRNGKey(0)
    src = jax.random.randint(key, (32, 50), 0, config.src_vocab_size)
    tgt = jax.random.randint(key, (32, 50), 0, config.tgt_vocab_size)
    mask = jax.random.randint(key, (32, 50), 0, 2).astype(bool)
    
    # Initialize model parameters
    variables = transformer.init(key, src, tgt, mask=mask, training=True)
    
    # Forward pass
    logits = transformer.apply(variables, src, tgt, mask=mask, training=False)
    
    # Check output shape
    assert logits.shape == (32, 50, config.tgt_vocab_size), "Output shape mismatch."
    
    print("TransformerBase model test passed.")

if __name__ == "__main__":
    test_transformer_base()
