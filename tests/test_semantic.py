"""Tests for semantic search and hybrid RRF retrieval."""

import pytest
from src.retrieval.semantic import SemanticSearchEngine
from src.retrieval.search import hybrid_search, search


@pytest.fixture
def sample_records():
    return [
        {
            "name": "login_handler",
            "kind": "function",
            "file": "auth.py",
            "start_line": 1,
            "end_line": 20,
            "docstring": "Authenticates user credentials and issues JWT token",
            "code": "def login_handler(username, password):\n    validate_user(username, password)",
        },
        {
            "name": "calculate_cart_total",
            "kind": "function",
            "file": "billing.py",
            "start_line": 30,
            "end_line": 50,
            "docstring": "Computes invoice balance including taxes and shipping",
            "code": "def calculate_cart_total(items):\n    return sum(i.price for i in items)",
        },
        {
            "name": "connect_database",
            "kind": "function",
            "file": "db.py",
            "start_line": 5,
            "end_line": 15,
            "docstring": "Initializes PostgreSQL connection pool",
            "code": "def connect_database():\n    return create_pool()",
        },
    ]


def test_semantic_search_engine_basic(sample_records):
    engine = SemanticSearchEngine(sample_records, k_dimensions=8)
    assert len(engine.vocab) > 0
    assert engine.doc_vectors.shape[0] == 3

    # Query with semantic term in docstring ("credentials")
    results = engine.search("credentials and passwords", top_k=2)
    assert len(results) >= 1
    top_score, top_rec = results[0]
    assert top_rec["file"] == "auth.py"
    assert top_score > 0.0


def test_semantic_search_empty():
    engine = SemanticSearchEngine([], k_dimensions=8)
    assert engine.search("anything") == []

    engine_with_docs = SemanticSearchEngine([{"name": "f", "file": "f.py", "kind": "f", "docstring": "", "code": "pass"}])
    assert engine_with_docs.search("") == []


def test_hybrid_search_rrf(sample_records):
    results = hybrid_search(
        "user authentication token",
        records=sample_records,
        top_k=2,
    )
    assert len(results) >= 1
    top_score, top_rec = results[0]
    assert top_rec["file"] == "auth.py"
    # Fused RRF score is > 0
    assert top_score > 0.0


def test_hybrid_search_flag(sample_records):
    # Testing search() with hybrid=True parameter
    results = search(
        "invoice and taxes",
        records=sample_records,
        top_k=2,
        hybrid=True,
    )
    assert len(results) >= 1
    assert results[0][1]["file"] == "billing.py"
