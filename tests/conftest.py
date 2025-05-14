"""
Pytest configuration file for the Howl Python application tests.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the app directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables needed for tests."""
    with patch.dict(os.environ, {
        "GEMINI_API_KEY": "fake-api-key",
        "HF_TOKEN": "fake-hf-token"
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

