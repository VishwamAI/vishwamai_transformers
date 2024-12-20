import random
import nltk
from nltk.corpus import wordnet
from typing import List, Union, Optional

class TextAugmenter:
    """
    A class for performing various text augmentation techniques.
    """
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n random words in the text with their synonyms.
        
        Args:
            text (str): Input text to augment
            n (int): Number of words to replace
            
        Returns:
            str: Augmented text
        """
        words = text.split()
        if len(words) <= n:
            return text
            
        new_words = words.copy()
        random_word_list = list(set([word for word in words if len(word) > 3]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = []
            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    if lemma.name() != random_word:
                        synonyms.append(lemma.name())
            
            if len(synonyms) >= 1:
                synonym = random.choice(list(set(synonyms)))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            
            if num_replaced >= n:
                break
                
        return ' '.join(new_words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words from the text with probability p.
        
        Args:
            text (str): Input text to augment
            p (float): Probability of deletion for each word
            
        Returns:
            str: Augmented text
        """
        words = text.split()
        if len(words) == 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
                
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            new_words.append(words[rand_int])
            
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap the positions of n pairs of words in the text.
        
        Args:
            text (str): Input text to augment
            n (int): Number of pairs to swap
            
        Returns:
            str: Augmented text
        """
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        return ' '.join(new_words)
    
    def back_translation(self, text: str, 
                        intermediate_lang: str = 'fr',
                        translator = None) -> str:
        """
        Perform back translation using an external translation service.
        Note: Requires a translation service to be passed in.
        
        Args:
            text (str): Input text to augment
            intermediate_lang (str): Intermediate language code
            translator: Translation service instance
            
        Returns:
            str: Augmented text
        """
        if translator is None:
            return text
            
        try:
            # Translate to intermediate language
            intermediate = translator.translate(text, dest=intermediate_lang).text
            # Translate back to original language
            back_translated = translator.translate(intermediate, dest='en').text
            return back_translated
        except:
            return text
    
    def augment(self, text: str, 
                techniques: List[str] = ['synonym', 'deletion', 'swap'],
                n_per_technique: int = 1) -> List[str]:
        """
        Apply multiple augmentation techniques to generate multiple variants.
        
        Args:
            text (str): Input text to augment
            techniques (List[str]): List of techniques to apply
            n_per_technique (int): Number of augmentations per technique
            
        Returns:
            List[str]: List of augmented texts
        """
        augmented_texts = []
        
        for _ in range(n_per_technique):
            if 'synonym' in techniques:
                augmented_texts.append(self.synonym_replacement(text))
            if 'deletion' in techniques:
                augmented_texts.append(self.random_deletion(text))
            if 'swap' in techniques:
                augmented_texts.append(self.random_swap(text))
                
        return augmented_texts
