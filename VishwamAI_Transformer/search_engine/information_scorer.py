import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup

class InformationScorer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def preprocess_text(self, text):
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])
        return text
    
    def score(self, query, retrieved_data):
        # Preprocess query
        query = self.preprocess_text(query)
        
        # Preprocess retrieved documents
        documents = []
        for doc in retrieved_data:
            if 'content' in doc:
                content = self.preprocess_text(doc['content'])
                documents.append(content)
            else:
                documents.append('')
        
        # Fit and transform the vectorizer
        if not documents:
            return []
        
        doc_vectors = self.vectorizer.fit_transform(documents)
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        # Create a list of (score, document) pairs
        scored_data = list(zip(similarities, retrieved_data))
        
        # Sort the list based on scores in descending order
        scored_data.sort(reverse=True, key=lambda x: x[0])
        
        return scored_data

if __name__ == "__main__":
    scorer = InformationScorer()
    query = "Latest AI research papers"
    retrieved_data = [
        {'content': "Recent advancements in AI have focused on deep learning."},
        {'content': "AI is changing the world."},
        {'content': "Machine learning is a subset of AI."},
        {'content': "This document is not related to AI."},
    ]
    ranked_data = scorer.score(query, retrieved_data)
    for score, doc in ranked_data:
        print(f"Score: {score:.4f}, Content: {doc['content']}")