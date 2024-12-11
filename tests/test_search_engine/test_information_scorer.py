import pytest
from VishwamAI_Transformer.search_engine.information_scorer import InformationScorer
from bs4 import BeautifulSoup

@pytest.fixture
def scorer():
    return InformationScorer()

def test_score(scorer):
    query = "machine learning"
    retrieved_data = [
        {'content': "Advancements in deep learning techniques."},
        {'content': "Machine learning algorithms are diverse."},
        {'content': "This document is unrelated."},
    ]
    ranked_data = scorer.score(query, retrieved_data)
    assert len(ranked_data) == 3
    assert ranked_data[0][0] > ranked_data[1][0] > ranked_data[2][0]

def test_empty_retrieved_data(scorer):
    query = "empty data"
    retrieved_data = []
    ranked_data = scorer.score(query, retrieved_data)
    assert ranked_data == []

def test_preprocess_text(scorer):
    text = "<p>Sample text with <b>HTML</b> tags and punctuation!</p>"
    cleaned_text = scorer.preprocess_text(text)
    assert cleaned_text == "sample text with html tags and punctuation"