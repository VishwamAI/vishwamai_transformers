import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import Scope

class PositionalEncoding(nn.Module):
    """
    Positional encoding module for Transformer models.
    
    Args:
        max_len: Maximum sequence length.
        d_model: Embedding dimension.
    """
    max_len: int
    d_model: int
    
    def setup(self):
        # Compute the positional encoding once
        position = jnp.arange(self.max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2, dtype=jnp.float32) * (-jnp.log(10000.0) / self.d_model))
        pe = jnp.zeros((self.max_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        # Declare it as a constant
        self.declare('pe', pe)
    
    def __call__(self, x):
        """
        Add positional encoding to the input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_length, d_model)
        
        Returns:
            Output embeddings with positional encoding added.
        """
        seq_length = x.shape[1]
        pe = self.variables['constants']['pe']
        return x + pe[:seq_length]

def test_positional_encoding():
    """
    Test function to verify the correctness of PositionalEncoding module.
    """
    max_len = 1000
    d_model = 512
    pe_module = PositionalEncoding(max_len=max_len, d_model=d_model)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (32, 20, d_model))
    y = pe_module(x)
    assert y.shape == (32, 20, d_model), "Output shape mismatch."
    
    # Check positional encoding for position 0
    pe = pe_module.variables['constants']['pe']
    assert jnp.allclose(pe[0, 0::2], jnp.zeros((d_model//2,))), "Even dimensions not zero for pos=0."
    assert jnp.allclose(pe[0, 1::2], jnp.ones((d_model//2,))), "Odd dimensions not one for pos=0."
    
    # Additional checks can be added for other positions as needed

if __name__ == "__main__":
    test_positional_encoding()
    print("PositionalEncoding module test passed.")