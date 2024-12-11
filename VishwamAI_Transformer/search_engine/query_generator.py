from spacy import load
from yake import KeywordExtractor
import re

class QueryGenerator:
    def __init__(self):
        self.nlp = load('en_core_web_sm')
        # Create extractors with different n-gram sizes
        self.unigram_extractor = KeywordExtractor(lan="en", n=1, top=5)
        self.bigram_extractor = KeywordExtractor(lan="en", n=2, top=5)

    def preprocess_text(self, text):
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return text

    def generate_query(self, input_text, max_queries=5, max_query_length=100):
        if not isinstance(input_text, str) or input_text.strip() == '':
            return [""]
            
        input_text = self.preprocess_text(input_text)
        if not input_text.strip():
            return [""]
            
        # Get named entities
        doc = self.nlp(input_text)
        entities = [ent.text.lower() for ent in doc.ents]
        
        # Extract keywords
        unigrams = [kw[0] for kw in self.unigram_extractor.extract_keywords(input_text)]
        bigrams = [kw[0] for kw in self.bigram_extractor.extract_keywords(input_text)]
        
        # Combine all potential queries
        queries = []
        queries.extend(bigrams)  # Add bigrams first for priority
        queries.extend(unigrams)
        
        # Add entity-based queries
        for entity in entities:
            queries.extend([f"about {entity}", f"information on {entity}"])
            
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = [q for q in queries if not (q in seen or seen.add(q))]
        
        # Trim queries to max length
        trimmed_queries = [q[:max_query_length] for q in unique_queries]
        
        # Return limited number of queries
        return trimmed_queries[:max_queries]