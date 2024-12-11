import re
from typing import List, Dict

class KnowledgeInjector:
    def __init__(self, tokenizer, max_seq_length, top_k=3):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.top_k = top_k

    def inject_knowledge(self, input_text, retrieved_data):
        if not retrieved_data:
            encoding = self.tokenizer.encode_plus(
                input_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
            return encoding
        
        selected_texts = self.select_relevant_texts(retrieved_data)
        enriched_input = self.modify_input(input_text, selected_texts)
        encoding = self.tokenizer.encode_plus(
            enriched_input,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoding

    def select_relevant_texts(self, retrieved_data):
        try:
            retrieved_data_sorted = sorted(
                retrieved_data,
                key=lambda x: x.get('score', 0),
                reverse=True
            )
            selected_texts = [
                self.clean_text(doc.get('content', ''))
                for doc in retrieved_data_sorted[:self.top_k]
            ]
        except TypeError:
            selected_texts = []
        return selected_texts

    def modify_input(self, input_text, selected_texts):
        if not selected_texts:
            return input_text
        injected_text = input_text + ' ' + ' '.join(selected_texts)
        return injected_text

    def clean_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

def generate_response(input_sequence, model, tokenizer, knowledge_injector, retrieved_data):
    enriched_input = knowledge_injector.inject_knowledge(input_sequence, retrieved_data)
    response = model.generate(enriched_input['input_ids'], attention_mask=enriched_input['attention_mask'])
    return response