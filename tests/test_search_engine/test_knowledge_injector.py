import pytest
from VishwamAI_Transformer.search_engine.knowledge_injector import KnowledgeInjector
from transformers import BertTokenizer

@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

@pytest.fixture
def knowledge_injector(tokenizer):
    return KnowledgeInjector(tokenizer, max_seq_length=128, top_k=2)

def test_inject_knowledge(knowledge_injector, tokenizer):
    input_text = "What are the health benefits of spinach?"
    retrieved_data = [
        {'score': 0.9, 'content': "Spinach is rich in iron and vitamins."},
        {'score': 0.8, 'content': "It also contains antioxidants."},
        {'score': 0.1, 'content': "Unrelated content."},
    ]
    encoding = knowledge_injector.inject_knowledge(input_text, retrieved_data)
    assert 'input_ids' in encoding
    assert 'attention_mask' in encoding
    assert len(encoding['input_ids'][0]) == 128

def test_no_retrieved_data(knowledge_injector, tokenizer):
    input_text = "No retrieved data available."
    encoding = knowledge_injector.inject_knowledge(input_text, [])
    assert 'input_ids' in encoding
    assert 'attention_mask' in encoding
    assert len(encoding['input_ids'][0]) == 128

def test_max_seq_length(knowledge_injector, tokenizer):
    input_text = "a" * 200
    retrieved_data = [
        {'score': 0.9, 'content': "b" * 200},
        {'score': 0.8, 'content': "c" * 200},
    ]
    encoding = knowledge_injector.inject_knowledge(input_text, retrieved_data)
    assert len(encoding['input_ids'][0]) == 128