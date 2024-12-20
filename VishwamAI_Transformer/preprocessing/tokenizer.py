import sentencepiece as spm
from typing import List, Dict, Union, Optional
from pathlib import Path
import os

class SentencePieceTokenizer:
    """
    A wrapper class for SentencePiece tokenizer with additional functionality
    for transformer models.
    """
    
    def __init__(self, 
                 vocab_size: int = 32000,
                 model_type: str = "bpe",
                 special_tokens: Dict[str, str] = None):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size (int): Vocabulary size
            model_type (str): Model type ('bpe' or 'unigram')
            special_tokens (Dict[str, str]): Dictionary of special tokens
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        
        # Define special tokens
        self.special_tokens = {
            'PAD': '[PAD]',
            'UNK': '[UNK]',
            'CLS': '[CLS]',
            'SEP': '[SEP]',
            'MASK': '[MASK]'
        }
        if special_tokens:
            self.special_tokens.update(special_tokens)
            
        self.sp_model = None
        
    def train(self, 
              texts: Union[List[str], str],
              model_prefix: str,
              min_freq: int = 2,
              character_coverage: float = 0.9995) -> None:
        """
        Train the SentencePiece tokenizer.
        
        Args:
            texts (Union[List[str], str]): Training texts or path to text file
            model_prefix (str): Prefix for saving model files
            min_freq (int): Minimum frequency for tokens
            character_coverage (float): Character coverage (important for non-English text)
        """
        # If texts is a list, write to temporary file
        if isinstance(texts, list):
            temp_file = f"{model_prefix}_temp.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(f"{text}\n")
            train_file = temp_file
        else:
            train_file = texts
            
        # Create training command
        train_args = {
            'input': train_file,
            'model_prefix': model_prefix,
            'vocab_size': self.vocab_size,
            'character_coverage': character_coverage,
            'model_type': self.model_type,
            'min_frequency': min_freq,
            'pad_id': 0,
            'unk_id': 1,
            'bos_id': 2,
            'eos_id': 3,
            'user_defined_symbols': list(self.special_tokens.values()),
            'input_sentence_size': 1000000,
            'shuffle_input_sentence': True
        }
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            ' '.join([f'--{k}={v}' for k, v in train_args.items()])
        )
        
        # Load the trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{model_prefix}.model")
        
        # Clean up temporary file if created
        if isinstance(texts, list) and os.path.exists(temp_file):
            os.remove(temp_file)
            
    def encode(self, 
               text: Union[str, List[str]], 
               add_special_tokens: bool = True) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token ids.
        
        Args:
            text (Union[str, List[str]]): Input text or list of texts
            add_special_tokens (bool): Whether to add special tokens
            
        Returns:
            Union[List[int], List[List[int]]]: Encoded token ids
        """
        if self.sp_model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
            
        if isinstance(text, str):
            text = [text]
            
        encoded = []
        for t in text:
            ids = self.sp_model.encode_as_ids(t)
            if add_special_tokens:
                cls_id = self.sp_model.piece_to_id(self.special_tokens['CLS'])
                sep_id = self.sp_model.piece_to_id(self.special_tokens['SEP'])
                ids = [cls_id] + ids + [sep_id]
            encoded.append(ids)
            
        return encoded[0] if len(encoded) == 1 else encoded
    
    def decode(self, 
               ids: Union[List[int], List[List[int]]], 
               remove_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token ids back to text.
        
        Args:
            ids (Union[List[int], List[List[int]]]): Token ids
            remove_special_tokens (bool): Whether to remove special tokens
            
        Returns:
            Union[str, List[str]]: Decoded text
        """
        if self.sp_model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
            
        if not isinstance(ids[0], list):
            ids = [ids]
            
        decoded = []
        special_ids = [self.sp_model.piece_to_id(token) for token in self.special_tokens.values()]
        
        for sequence in ids:
            if remove_special_tokens:
                sequence = [id for id in sequence if id not in special_ids]
            text = self.sp_model.decode_ids(sequence)
            decoded.append(text)
            
        return decoded[0] if len(decoded) == 1 else decoded
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer model and configuration.
        
        Args:
            path (str): Directory path to save the tokenizer
        """
        if self.sp_model is None:
            raise ValueError("No model to save. Train or load a model first.")
            
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save the model and vocabulary files
        self.sp_model.save(str(path / "tokenizer.model"))
        
    def load(self, path: str) -> None:
        """
        Load a trained tokenizer model.
        
        Args:
            path (str): Path to the directory containing the tokenizer files
        """
        path = Path(path)
        model_path = path / "tokenizer.model"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No model file found at {model_path}")
            
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(str(model_path))
        
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size.
        
        Returns:
            int: Vocabulary size
        """
        return self.sp_model.get_piece_size() if self.sp_model else 0
    
    def token_to_id(self, token: str) -> int:
        """
        Convert token to id.
        
        Args:
            token (str): Input token
            
        Returns:
            int: Token id
        """
        return self.sp_model.piece_to_id(token) if self.sp_model else None
    
    def id_to_token(self, id: int) -> str:
        """
        Convert id to token.
        
        Args:
            id (int): Token id
            
        Returns:
            str: Token
        """
        return self.sp_model.id_to_piece(id) if self.sp_model else None
