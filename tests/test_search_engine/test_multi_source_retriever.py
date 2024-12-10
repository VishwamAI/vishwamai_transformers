import jax
import jax.numpy as jnp
from flax import linen as nn
from VishwamAI_Transformer.search_engine.multi_source_retriever import MultiSourceInformationRetriever

def test_multi_source_information_retriever():
    """
    Test function to verify the MultiSourceInformationRetriever module.
    """
    num_sources = 3
    d_model = 512
    num_heads = 8
    head_dim = 64
    num_layers = 6
    dropout_rate = 0.1
    
    retriever = MultiSourceInformationRetriever(
        num_sources=num_sources,
        d_model=d_model,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    
    key = jax.random.PRNGKey(0)
    input_sequences = [jax.random.randint(key, (32, 20), 0, 10000) for _ in range(num_sources)]
    positions = jax.random.randint(key, (32, 20), 0, 1000)
    
    variables = retriever.init(key, input_sequences, positions, training=True)
    output = retriever.apply(variables, input_sequences, positions, training=False)
    
    assert output.shape == (32, 20, d_model), "Output shape mismatch."
    
    print("MultiSourceInformationRetriever module test passed.")

if __name__ == "__main__":
    test_multi_source_information_retriever()
