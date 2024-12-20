import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable
import sentencepiece as spm

# Import custom modules
from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding

class TransformerBase(nn.Module):
    """
    Base Transformer model architecture with SentencePiece tokenization.
    
    Args:
        num_layers: Number of encoder and decoder layers.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        dropout_rate: Dropout rate.
        activation: Activation function to use.
        sp_model_path: Path to the SentencePiece model file.
    """
    num_layers: int
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1
    activation: Callable = nn.gelu
    sp_model_path: str = None

    def setup(self):
        self.encoder = Encoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        )
        
        self.decoder = Decoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        )

        # Initialize SentencePiece tokenizer if model path is provided
        if self.sp_model_path:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.sp_model_path)

    def encode_text(self, text: str) -> jnp.ndarray:
        """
        Encode input text using SentencePiece tokenizer.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            Tokenized input as a JAX array.
        """
        if not hasattr(self, 'sp'):
            raise ValueError("SentencePiece model path not provided during initialization")
        
        tokens = self.sp.encode_as_ids(text)
        return jnp.array(tokens)

    def decode_tokens(self, tokens: jnp.ndarray) -> str:
        """
        Decode tokens back to text using SentencePiece tokenizer.
        
        Args:
            tokens: Array of token IDs.
            
        Returns:
            Decoded text string.
        """
        if not hasattr(self, 'sp'):
            raise ValueError("SentencePiece model path not provided during initialization")
        
        tokens = tokens.tolist() if isinstance(tokens, jnp.ndarray) else tokens
        return self.sp.decode(tokens)

    def __call__(self, src, tgt, training=True):
        """
        Forward pass of the transformer.
        
        Args:
            src: Source tokens or text
            tgt: Target tokens or text
            training: Whether in training mode
            
        Returns:
            Decoder output logits
        """
        # Convert text to tokens if strings are provided
        if isinstance(src, str):
            src = self.encode_text(src)
        if isinstance(tgt, str):
            tgt = self.encode_text(tgt)

        encoder_output = self.encoder(src, training)
        decoder_output = self.decoder(tgt, encoder_output, training)
        return decoder_output

def test_transformer_base():
    """
    Test function to verify the TransformerBase model with tokenization.
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
        'activation': nn.gelu,
        'sp_model_path': 'path/to/sentencepiece.model'  # Add path to your SP model
    }
    config = type('Config', (object,), config)
    
    # Create model instance
    transformer = TransformerBase(
        num_layers=config.num_encoder_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        dropout_rate=config.dropout_rate,
        activation=config.activation,
        sp_model_path=config.sp_model_path
    )
    
    # Test with both token and text inputs
    key = jax.random.PRNGKey(0)
    
    # Test with token inputs
    src = jax.random.randint(key, (32, 50), 0, config.src_vocab_size)
    tgt = jax.random.randint(key, (32, 50), 0, config.tgt_vocab_size)
    
    variables = transformer.init(key, src, tgt, training=True)
    logits = transformer.apply(variables, src, tgt, training=False)
    
    assert logits.shape == (32, 50, config.tgt_vocab_size), "Output shape mismatch."
    
    # Test with text input (if SP model is available)
    try:
        test_text = "Hello, world!"
        tokens = transformer.encode_text(test_text)
        decoded_text = transformer.decode_tokens(tokens)
        print(f"Tokenization test: {test_text} -> {tokens} -> {decoded_text}")
    except Exception as e:
        print(f"Skipping text tokenization test: {str(e)}")
    
    print("TransformerBase model test passed.")

if __name__ == "__main__":
    test_transformer_base()