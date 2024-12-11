import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Optional
from VishwamAI_Transformer.models.positional_encoding import PositionalEncoding
from VishwamAI_Transformer.models.transformer_base import TransformerBase
class RealTimeSearchQueryGenerator(nn.Module):
    """
    Real-time search query generation module.
    
    Args:
        vocab_size: Size of the vocabulary.
        d_model: Embedding dimension.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        num_layers: Number of layers in the transformer.
        dropout_rate: Dropout rate for attention layers.
    """
    vocab_size: int
    d_model: int
    num_heads: int
    head_dim: int
    num_layers: int
    dropout_rate: float
    
    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)
        self.positional_encoding = PositionalEncoding(max_len=1000, d_model=self.d_model)
        self.transformer = TransformerBase(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            activation=nn.gelu
        )
        self.output_layer = nn.Dense(features=self.vocab_size)
    
    def __call__(self, input_sequence, positions, training=True):
        """
        Generate search query in real-time.
        
        Args:
            input_sequence: Input sequence of shape (batch_size, seq_length)
            positions: Positions of shape (batch_size, seq_length)
            training: Boolean indicating whether the model is in training mode
        
        Returns:
            Output sequence of shape (batch_size, seq_length, vocab_size)
        """
        x = self.embedding(input_sequence)
        x = self.positional_encoding(x, positions)
        x = self.transformer(x, x, training)
        output = self.output_layer(x)
        return output

def test_real_time_search_query_generator():
    """
    Test function to verify the RealTimeSearchQueryGenerator module.
    """
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    head_dim = 64
    num_layers = 6
    dropout_rate = 0.1
    
    query_generator = RealTimeSearchQueryGenerator(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    
    key = jax.random.PRNGKey(0)
    input_sequence = jax.random.randint(key, (32, 20), 0, vocab_size)
    positions = jax.random.randint(key, (32, 20), 0, 1000)
    
    variables = query_generator.init(key, input_sequence, positions, training=True)
    output = query_generator.apply(variables, input_sequence, positions, training=False)
    
    assert output.shape == (32, 20, vocab_size), "Output shape mismatch."
    
    print("RealTimeSearchQueryGenerator module test passed.")

if __name__ == "__main__":
    test_real_time_search_query_generator()
