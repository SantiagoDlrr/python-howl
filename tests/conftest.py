"""
Pytest configuration file for the Howl Python application tests.
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

# Add the app directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables needed for tests."""
    with patch.dict(os.environ, {
        "GEMINI_API_KEY": "fake-api-key",
        "HF_TOKEN": "fake-hf-token",
        "PINECONE_KEY": "fake-pinecone-key"
    }):
        yield

# Create a fixture to mock the OCI configuration
@pytest.fixture(autouse=True)  # autouse=True makes this run for all tests
def mock_oci_config():
    """Mock the OCI configuration."""
    with patch('oci.config.from_file') as mock_from_file:
        mock_from_file.return_value = {
            "user": "fake-user",
            "fingerprint": "fake-fingerprint",
            "tenancy": "fake-tenancy",
            "region": "fake-region",
            "key_file": "fake-key-file"
        }
        yield mock_from_file

# Create a fixture to mock the OCI client
@pytest.fixture(autouse=True)  # autouse=True makes this run for all tests
def mock_oci_client():
    """Mock the OCI Language client."""
    with patch("oci.ai_language.AIServiceLanguageClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance

# Create a fixture to mock the Embeddings class
@pytest.fixture
def mock_embeddings():
    """Mock the Embeddings class."""
    with patch("app.embeddings.embeddings.Embeddings") as mock_embeddings_class:
        mock_instance = MagicMock()
        mock_instance.generate_embeddings.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        mock_embeddings_class.return_value = mock_instance
        yield mock_instance

# Create a fixture to mock the database query function
@pytest.fixture
def mock_db_query():
    """Mock the database query function."""
    with patch("app.db.query") as mock_query:
        mock_query.return_value = AsyncMock()
        yield mock_query

# Create a fixture to mock the Gemini model
@pytest.fixture
def mock_gemini_model():
    """Mock the Gemini model."""
    with patch("google.generativeai.GenerativeModel") as mock_model_class:
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a test response from the Gemini model."
        mock_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_instance
        yield mock_instance

# Create a fixture for sample RAG chat data
@pytest.fixture
def sample_rag_chat_data():
    """Sample data for RAG chat tests."""
    return {
        "question": "What was the customer's issue with their order?",
        "call_ids": ["call-123"],
        "model_name": "gemini-1.5-flash",
        "api_key": "fake-api-key",
        "request_id": "test-request-id",
        "response": {
            "answer": "The customer's order was delayed and hadn't arrived when expected.",
            "sources": [
                {
                    "call_id": "call-123",
                    "text": "CUSTOMER: I'm having trouble with my recent order. It was supposed to arrive yesterday but I haven't received it yet.",
                    "score": 0.9
                }
            ]
        }
    }

