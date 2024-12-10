import jax
import jax.numpy as jnp
from flax import linen as nn
from VishwamAI_Transformer.models.transformer_base import TransformerBase

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
    transformer = TransformerBase(
        num_layers=config.num_encoder_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        dropout_rate=config.dropout_rate,
        activation=config.activation
    )
    
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
