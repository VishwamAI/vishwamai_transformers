import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import Scope

class DynamicPositionalEncoding(nn.Module):
    """
    Dynamic positional encoding module for Transformer models.
    
    Args:
        d_model: Embedding dimension.
    """
    d_model: int
    
    @nn.compact
    def __call__(self, x, positions):
        """
        Add dynamic positional encoding to the input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_length, d_model)
            positions: Positions of shape (batch_size, seq_length)
        
        Returns:
            Output embeddings with dynamic positional encoding added.
        """
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2, dtype=jnp.float32) * (-jnp.log(10000.0) / self.d_model))
        pe = jnp.zeros((positions.shape[0], positions.shape[1], self.d_model))
        pe = pe.at[:, :, 0::2].set(jnp.sin(positions * div_term))
        pe = pe.at[:, :, 1::2].set(jnp.cos(positions * div_term))
        return x + pe

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
        self.dynamic_pe = DynamicPositionalEncoding(d_model=self.d_model)
    
    def __call__(self, x, positions):
        """
        Add positional encoding to the input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_length, d_model)
            positions: Positions of shape (batch_size, seq_length)
        
        Returns:
            Output embeddings with positional encoding added.
        """
        return self.dynamic_pe(x, positions)

def test_positional_encoding():
    """
    Test function to verify the correctness of PositionalEncoding module.
    """
    max_len = 1000
    d_model = 512
    pe_module = PositionalEncoding(max_len=max_len, d_model=d_model)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (32, 20, d_model))
    positions = jax.random.randint(key, (32, 20), 0, max_len)
    y = pe_module(x, positions)
    assert y.shape == (32, 20, d_model), "Output shape mismatch."
    
    # Additional checks can be added for other positions as needed

if __name__ == "__main__":
    test_positional_encoding()
    print("PositionalEncoding module test passed.")