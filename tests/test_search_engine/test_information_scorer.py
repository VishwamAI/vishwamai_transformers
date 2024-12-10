import jax
import jax.numpy as jnp
from flax import linen as nn
from VishwamAI_Transformer.search_engine.information_scorer import ContextualInformationScorer

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
