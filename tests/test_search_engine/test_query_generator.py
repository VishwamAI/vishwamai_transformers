import pytest
from VishwamAI_Transformer.search_engine.query_generator import QueryGenerator

@pytest.fixture
def query_generator():
    return QueryGenerator()

def test_generate_query(query_generator):
    input_text = "What are the health benefits of eating spinach?"
    queries = query_generator.generate_query(input_text, max_queries=3)
    assert len(queries) == 3
    assert any("health benefits" in query for query in queries)
    assert any("spinach" in query for query in queries)  # This should now pass

def test_empty_input(query_generator):
    input_text = ""
    queries = query_generator.generate_query(input_text)
    assert queries == [""]

def test_invalid_input(query_generator):
    input_text = "!!!"
    queries = query_generator.generate_query(input_text)
    assert queries == [""]

def test_unique_queries(query_generator):
    input_text = "I want to learn programming. Programming is fun."
    queries = query_generator.generate_query(input_text, max_queries=5)
    assert len(queries) == 5
    assert len(set(queries)) == len(queries)