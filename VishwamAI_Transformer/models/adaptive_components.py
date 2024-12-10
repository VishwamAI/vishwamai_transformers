import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional

class AdaptiveNormalization(nn.Module):
    features: int
    epsilon: float = 1e-5
    use_bias: bool = True
    use_scale: bool = True
    
    @nn.compact
    def __call__(self, x, context=None):
        # Compute mean and variance
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        
        if context is not None:
            # Generate adaptive scaling parameters
            context_scaling = nn.Dense(features=self.features * 2)(context)
            shift, scale = jnp.split(context_scaling, 2, axis=-1)
            mean += shift
            var *= (1 + scale)
        
        # Normalize
        x_norm = (x - mean) / jnp.sqrt(var + self.epsilon)
        
        # Learnable affine transformation
        if self.use_scale:
            scale_param = self.param('scale', nn.initializers.ones, (self.features,))
            x_norm *= scale_param
        
        if self.use_bias:
            bias_param = self.param('bias', nn.initializers.zeros, (self.features,))
            x_norm += bias_param
        
        return x_norm

class DynamicFFN(nn.Module):
    initial_features: int
    dropout_rate: float = 0.1
    activation: Callable = nn.gelu
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Dynamic feature generation
        feature_multiplier = self.param('feature_multiplier', nn.initializers.ones, (1,))
        current_features = int(self.initial_features * jnp.abs(feature_multiplier[0]))
        
        # First projection with adaptive width
        x = nn.Dense(current_features)(x)
        x = self.activation(x)
        
        # Adaptive dropout
        if training:
            x = nn.Dropout(self.dropout_rate)(x)
        
        # Residual gating mechanism
        gate = nn.Dense(current_features, use_bias=False)(x)
        gate = jax.nn.sigmoid(gate)
        
        # Second projection with gated residual
        x = nn.Dense(self.initial_features)(x)
        x = x * gate
        
        return x

class AdaptiveAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, query, key, value, mask=None, training: bool = False):
        # Dynamic head generation
        head_selector = self.param('head_selector', nn.initializers.uniform(), (self.num_heads,))
        head_weights = jax.nn.softmax(head_selector)
        
        # Adaptive multi-head attention
        def single_head_attention(q, k, v, head_weight):
            # Scaled dot-product attention
            attention_scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.head_dim)
            if mask is not None:
                attention_scores = jnp.where(mask, attention_scores, float('-inf'))
            attention_probs = nn.softmax(attention_scores, axis=-1)
            if training:
                attention_probs = nn.Dropout(self.dropout_rate)(attention_probs)
            head_output = jnp.matmul(attention_probs, v)
            return head_output * head_weight
        
        # Compute each head
        heads = []
        for i in range(self.num_heads):
            head_weight = head_weights[i]
            head_q = nn.Dense(self.head_dim)(query)
            head_k = nn.Dense(self.head_dim)(key)
            head_v = nn.Dense(self.head_dim)(value)
            head_output = single_head_attention(head_q, head_k, head_v, head_weight)
            heads.append(head_output)
        
        # Combine heads
        combined_heads = jnp.concatenate(heads, axis=-1)
        output = nn.Dense(query.shape[-1])(combined_heads)
        return output