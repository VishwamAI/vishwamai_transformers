import jax
import jax.numpy as jnp
from flax import linen as nn
from VishwamAI_Transformer.search_engine.knowledge_injector import DynamicKnowledgeInjector

def test_dynamic_knowledge_injector():
    """
    Test function to verify the DynamicKnowledgeInjector module.
    """
    d_model = 512
    num_heads = 8
    head_dim = 64
    num_layers = 6
    dropout_rate = 0.1
    
    injector = DynamicKnowledgeInjector(
        d_model=d_model,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    
    key = jax.random.PRNGKey(0)
    input_sequence = jax.random.randint(key, (32, 20), 0, 10000)
    positions = jax.random.randint(key, (32, 20), 0, 1000)
    knowledge = jax.random.normal(key, (32, 20, d_model))
    
    variables = injector.init(key, input_sequence, positions, knowledge, training=True)
    output = injector.apply(variables, input_sequence, positions, knowledge, training=False)
    
    assert output.shape == (32, 20, d_model), "Output shape mismatch."
    
    print("DynamicKnowledgeInjector module test passed.")

if __name__ == "__main__":
    test_dynamic_knowledge_injector()
