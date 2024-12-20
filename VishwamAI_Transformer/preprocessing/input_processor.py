import re
import unicodedata
from typing import List, Dict, Union, Optional
from collections import Counter
import numpy as np
from .data_augmentation import TextAugmenter

class InputProcessor:
    """
    A class for processing and tokenizing input text for transformer models.
    """
    
    def __init__(self, 
                 vocab_size: int = 30000,
                 max_length: int = 512,
                 min_freq: int = 2,
                 special_tokens: Dict[str, str] = None):
        """
        Initialize the input processor.
        
        Args:
            vocab_size (int): Maximum size of vocabulary
            max_length (int): Maximum sequence length
            min_freq (int): Minimum frequency for a token to be included in vocab
            special_tokens (Dict[str, str]): Dictionary of special tokens
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_freq = min_freq
        
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
            
        # Initialize vocabulary
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Initialize augmenter
        self.augmenter = TextAugmenter()
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra whitespace, converting to lowercase,
        and performing unicode normalization.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from list of texts.
        
        Args:
            texts (List[str]): List of input texts
        """
        # Count word frequencies
        for text in texts:
            words = self.normalize_text(text).split()
            self.word_freq.update(words)
        
        # Add special tokens first
        vocab = list(self.special_tokens.values())
        
        # Add words that meet minimum frequency
        vocab.extend([word for word, freq in self.word_freq.most_common()
                     if freq >= self.min_freq])
        
        # Truncate vocabulary to maximum size
        vocab = vocab[:self.vocab_size]
        
        # Create word to index mappings
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to sequence of token indices.
        
        Args:
            text (str): Input text
            add_special_tokens (bool): Whether to add [CLS] and [SEP] tokens
            
        Returns:
            List[int]: List of token indices
        """
        # Normalize text
        text = self.normalize_text(text)
        
        # Convert words to indices
        tokens = []
        if add_special_tokens:
            tokens.append(self.word2idx[self.special_tokens['CLS']])
            
        for word in text.split():
            tokens.append(self.word2idx.get(word, self.word2idx[self.special_tokens['UNK']]))
            
        if add_special_tokens:
            tokens.append(self.word2idx[self.special_tokens['SEP']])
            
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Convert sequence of token indices back to text.
        
        Args:
            tokens (List[int]): List of token indices
            
        Returns:
            str: Decoded text
        """
        words = [self.idx2word.get(idx, self.special_tokens['UNK']) for idx in tokens]
        # Remove special tokens
        words = [word for word in words if word not in self.special_tokens.values()]
        return ' '.join(words)
    
    def pad_sequence(self, sequence: List[int]) -> List[int]:
        """
        Pad or truncate sequence to max_length.
        
        Args:
            sequence (List[int]): Input sequence
            
        Returns:
            List[int]: Padded sequence
        """
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        else:
            pad_token = self.word2idx[self.special_tokens['PAD']]
            return sequence + [pad_token] * (self.max_length - len(sequence))
    
    def create_attention_mask(self, sequence: List[int]) -> List[int]:
        """
        Create attention mask for padded sequence.
        
        Args:
            sequence (List[int]): Input sequence
            
        Returns:
            List[int]: Attention mask (1 for tokens, 0 for padding)
        """
        pad_token = self.word2idx[self.special_tokens['PAD']]
        return [1 if token != pad_token else 0 for token in sequence]
    
    def prepare_input(self, 
                     texts: Union[str, List[str]], 
                     add_special_tokens: bool = True) -> Dict[str, np.ndarray]:
        """
        Prepare input for transformer model.
        
        Args:
            texts (Union[str, List[str]]): Input text or list of texts
            add_special_tokens (bool): Whether to add special tokens
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing input_ids and attention_mask
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Encode all texts
        encoded = [self.encode(text, add_special_tokens) for text in texts]
        
        # Pad sequences
        padded = [self.pad_sequence(seq) for seq in encoded]
        
        # Create attention masks
        attention_masks = [self.create_attention_mask(seq) for seq in padded]
        
        return {
            'input_ids': np.array(padded),
            'attention_mask': np.array(attention_masks)
        }
    
    def augment_text(self, 
                     text: str, 
                     techniques: List[str] = ['synonym', 'deletion', 'swap'],
                     n_per_technique: int = 1) -> List[str]:
        """
        Augment input text using specified techniques.
        
        Args:
            text (str): Input text
            techniques (List[str]): List of augmentation techniques to apply
            n_per_technique (int): Number of augmentations per technique
            
        Returns:
            List[str]: List of augmented texts
        """
        return self.augmenter.augment(text, techniques, n_per_technique)

