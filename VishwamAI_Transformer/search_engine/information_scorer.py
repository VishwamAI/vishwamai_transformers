import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Optional
from VishwamAI_Transformer.models.positional_encoding import PositionalEncoding
from VishwamAI_Transformer.models.transformer_base import TransformerBase

class ContextualInformationScorer(nn.Module):
    """
    Contextual information scoring module.
    
    Args:
        d_model: Embedding dimension.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        num_layers: Number of layers in the transformer.
        dropout_rate: Dropout rate for attention layers.
    """
    d_model: int
    num_heads: int
    head_dim: int
    num_layers: int
    dropout_rate: float
    
    def setup(self):
        self.embedding = nn.Embed(num_embeddings=10000, features=self.d_model)
        self.positional_encoding = PositionalEncoding(max_len=1000, d_model=self.d_model)
        self.transformer = TransformerBase(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            activation=nn.gelu
        )
        self.output_layer = nn.Dense(features=1)
    
    def __call__(self, input_sequence, positions, training=True):
        """
        Score contextual information.
        
        Args:
            input_sequence: Input sequence of shape (batch_size, seq_length)
            positions: Positions of shape (batch_size, seq_length)
            training: Boolean indicating whether the model is in training mode
        
        Returns:
            Scores of shape (batch_size, seq_length, 1)
        """
        x = self.embedding(input_sequence)
        x = self.positional_encoding(x, positions)
        x = self.transformer(x, x, training)
        scores = self.output_layer(x)
        return scores

def test_contextual_information_scorer():
    """
    Test function to verify the ContextualInformationScorer module.
    """
    d_model = 512
    num_heads = 8
    head_dim = 64
    num_layers = 6
    dropout_rate = 0.1
    
    scorer = ContextualInformationScorer(
        d_model=d_model,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    
    key = jax.random.PRNGKey(0)
    input_sequence = jax.random.randint(key, (32, 20), 0, 10000)
    positions = jax.random.randint(key, (32, 20), 0, 1000)
    
    variables = scorer.init(key, input_sequence, positions, training=True)
    scores = scorer.apply(variables, input_sequence, positions, training=False)
    
    assert scores.shape == (32, 20, 1), "Output shape mismatch."
    
    print("ContextualInformationScorer module test passed.")

if __name__ == "__main__":
    test_contextual_information_scorer()
