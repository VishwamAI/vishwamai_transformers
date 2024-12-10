import jax
import jax.numpy as jnp
from flax import linen as nn
from VishwamAI_Transformer.search_engine.query_generator import RealTimeSearchQueryGenerator

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
