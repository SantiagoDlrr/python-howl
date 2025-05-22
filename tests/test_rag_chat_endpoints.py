"""
Unit tests for the RAG chat endpoints in main.py.
"""
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Import the FastAPI app and related models
from app.main import (
    app,
    RAGChatRequest,
    RAGChatStartRequest,
    RAGChatStartResponse,
    RAGChatStatusResponse
)

# Create a test client
client = TestClient(app)

# Sample data for tests
SAMPLE_QUESTION = "What was the customer's issue with their order?"
SAMPLE_CALL_IDS = ["call-123"]
SAMPLE_REQUEST_ID = "test-request-id"
SAMPLE_RESPONSE = {
    "answer": "The customer's order was delayed and hadn't arrived when expected.",
    "sources": [
        {
            "call_id": "call-123",
            "text": "CUSTOMER: I'm having trouble with my recent order. It was supposed to arrive yesterday but I haven't received it yet.",
            "score": 0.9
        }
    ]
}

class TestRagChatEndpoints:
    """Tests for the RAG chat endpoints in main.py."""

    @patch("app.main.rag_chat_function")
    def test_rag_chat_endpoint(self, mock_rag_chat_function):
        """Test the /rag_chat endpoint."""
        # Mock the rag_chat_function to return a sample response
        mock_rag_chat_function.return_value = SAMPLE_RESPONSE
        
        # Create a test request
        request_data = {
            "question": SAMPLE_QUESTION,
            "call_ids": SAMPLE_CALL_IDS
        }
        
        # Send a POST request to the endpoint
        response = client.post("/rag_chat", json=request_data)
        
        # Check the response
        assert response.status_code == 200
        assert response.json() == SAMPLE_RESPONSE
        mock_rag_chat_function.assert_called_once()
    
    def test_rag_chat_endpoint_empty_question(self):
        """Test the /rag_chat endpoint with an empty question."""
        # Create a test request with an empty question
        request_data = {
            "question": "",
            "call_ids": SAMPLE_CALL_IDS
        }
        
        # Send a POST request to the endpoint
        response = client.post("/rag_chat", json=request_data)
        
        # Check the response
        assert response.status_code == 400
        assert "Question cannot be empty" in response.json()["detail"]
    
    def test_rag_chat_endpoint_empty_call_ids(self):
        """Test the /rag_chat endpoint with empty call IDs."""
        # Create a test request with empty call IDs
        request_data = {
            "question": SAMPLE_QUESTION,
            "call_ids": []
        }
        
        # Send a POST request to the endpoint
        response = client.post("/rag_chat", json=request_data)
        
        # Check the response
        assert response.status_code == 400
        assert "At least one call ID must be provided" in response.json()["detail"]
    
    @patch("app.main.rag_chat_function")
    def test_rag_chat_endpoint_exception(self, mock_rag_chat_function):
        """Test the /rag_chat endpoint when an exception occurs."""
        # Mock the rag_chat_function to raise an exception
        mock_rag_chat_function.side_effect = Exception("Test exception")
        
        # Create a test request
        request_data = {
            "question": SAMPLE_QUESTION,
            "call_ids": SAMPLE_CALL_IDS
        }
        
        # Send a POST request to the endpoint
        response = client.post("/rag_chat", json=request_data)
        
        # Check the response
        assert response.status_code == 503
        assert "RAG chat service error" in response.json()["detail"]
    
    @patch("app.main.create_rag_chat_request")
    @patch("app.main.request_tracker.get_request_status")
    def test_start_rag_chat_endpoint(self, mock_get_request_status, mock_create_request):
        """Test the /rag_chat/start endpoint."""
        # Mock the create_rag_chat_request function to return a sample request ID
        mock_create_request.return_value = SAMPLE_REQUEST_ID
        
        # Mock the get_request_status function to return a sample status
        mock_get_request_status.return_value = {
            "status": "pending"
        }
        
        # Create a test request
        request_data = {
            "question": SAMPLE_QUESTION,
            "call_ids": SAMPLE_CALL_IDS
        }
        
        # Send a POST request to the endpoint
        response = client.post("/rag_chat/start", json=request_data)
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["request_id"] == SAMPLE_REQUEST_ID
        assert response.json()["status"] == "pending"
        mock_create_request.assert_called_once()
        mock_get_request_status.assert_called_once_with(SAMPLE_REQUEST_ID)
    
    def test_start_rag_chat_endpoint_empty_question(self):
        """Test the /rag_chat/start endpoint with an empty question."""
        # Create a test request with an empty question
        request_data = {
            "question": "",
            "call_ids": SAMPLE_CALL_IDS
        }
        
        # Send a POST request to the endpoint
        response = client.post("/rag_chat/start", json=request_data)
        
        # Check the response
        assert response.status_code == 400
        assert "Question cannot be empty" in response.json()["detail"]
    
    def test_start_rag_chat_endpoint_empty_call_ids(self):
        """Test the /rag_chat/start endpoint with empty call IDs."""
        # Create a test request with empty call IDs
        request_data = {
            "question": SAMPLE_QUESTION,
            "call_ids": []
        }
        
        # Send a POST request to the endpoint
        response = client.post("/rag_chat/start", json=request_data)
        
        # Check the response
        assert response.status_code == 400
        assert "At least one call ID must be provided" in response.json()["detail"]
    
    @patch("app.main.request_tracker.get_request_status")
    def test_get_rag_chat_status(self, mock_get_request_status):
        """Test the /rag_chat/status/{request_id} endpoint."""
        # Mock the get_request_status function to return a sample status
        mock_get_request_status.return_value = {
            "request_id": SAMPLE_REQUEST_ID,
            "status": "completed",
            "created_at": 1625097600.0,
            "updated_at": 1625097605.0,
            "result": SAMPLE_RESPONSE,
            "error": None
        }
        
        # Send a GET request to the endpoint
        response = client.get(f"/rag_chat/status/{SAMPLE_REQUEST_ID}")
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["request_id"] == SAMPLE_REQUEST_ID
        assert response.json()["status"] == "completed"
        assert response.json()["result"] == SAMPLE_RESPONSE
        mock_get_request_status.assert_called_once_with(SAMPLE_REQUEST_ID)
    
    @patch("app.main.request_tracker.get_request_status")
    def test_get_rag_chat_status_not_found(self, mock_get_request_status):
        """Test the /rag_chat/status/{request_id} endpoint when the request is not found."""
        # Mock the get_request_status function to return None
        mock_get_request_status.return_value = None
        
        # Send a GET request to the endpoint
        response = client.get(f"/rag_chat/status/{SAMPLE_REQUEST_ID}")
        
        # Check the response
        assert response.status_code == 404
        assert "Request not found" in response.json()["detail"]
        mock_get_request_status.assert_called_once_with(SAMPLE_REQUEST_ID)
