"""
Unit tests for the asynchronous processing functionality in the rag_chat.py module.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.rag_chat import process_rag_chat_request, create_rag_chat_request
from app.request_tracker import RequestStatus

class TestRagChatAsync:
    """Tests for the asynchronous processing functionality in the rag_chat.py module."""

    @patch("app.rag_chat.request_tracker")
    def test_create_rag_chat_request(self, mock_request_tracker):
        """Test creating a new RAG chat request."""
        # Mock the request_tracker.create_request method
        mock_request_tracker.create_request.return_value = "test-request-id"
        
        # Call the function
        request_id = create_rag_chat_request()
        
        # Check the result
        assert request_id == "test-request-id"
        mock_request_tracker.create_request.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.rag_chat.rag_chat")
    @patch("app.rag_chat.request_tracker")
    async def test_process_rag_chat_request_success(self, mock_request_tracker, mock_rag_chat, sample_rag_chat_data):
        """Test processing a RAG chat request successfully."""
        # Mock the rag_chat function
        mock_rag_chat.return_value = sample_rag_chat_data["response"]
        
        # Call the function
        await process_rag_chat_request(
            request_id=sample_rag_chat_data["request_id"],
            question=sample_rag_chat_data["question"],
            call_ids=sample_rag_chat_data["call_ids"],
            model_name=sample_rag_chat_data["model_name"],
            api_key=sample_rag_chat_data["api_key"]
        )
        
        # Check that the request status was updated correctly
        mock_request_tracker.update_request_status.assert_called_once_with(
            sample_rag_chat_data["request_id"],
            RequestStatus.PROCESSING
        )
        
        # Check that the rag_chat function was called with the correct arguments
        mock_rag_chat.assert_called_once_with(
            sample_rag_chat_data["question"],
            sample_rag_chat_data["call_ids"],
            sample_rag_chat_data["model_name"],
            sample_rag_chat_data["api_key"]
        )
        
        # Check that the request result was set correctly
        mock_request_tracker.set_request_result.assert_called_once_with(
            sample_rag_chat_data["request_id"],
            sample_rag_chat_data["response"]
        )
    
    @pytest.mark.asyncio
    @patch("app.rag_chat.rag_chat")
    @patch("app.rag_chat.request_tracker")
    async def test_process_rag_chat_request_error(self, mock_request_tracker, mock_rag_chat, sample_rag_chat_data):
        """Test processing a RAG chat request with an error."""
        # Mock the rag_chat function to raise an exception
        mock_rag_chat.side_effect = Exception("Test error")
        
        # Call the function
        await process_rag_chat_request(
            request_id=sample_rag_chat_data["request_id"],
            question=sample_rag_chat_data["question"],
            call_ids=sample_rag_chat_data["call_ids"],
            model_name=sample_rag_chat_data["model_name"],
            api_key=sample_rag_chat_data["api_key"]
        )
        
        # Check that the request status was updated correctly
        mock_request_tracker.update_request_status.assert_called_once_with(
            sample_rag_chat_data["request_id"],
            RequestStatus.PROCESSING
        )
        
        # Check that the rag_chat function was called with the correct arguments
        mock_rag_chat.assert_called_once_with(
            sample_rag_chat_data["question"],
            sample_rag_chat_data["call_ids"],
            sample_rag_chat_data["model_name"],
            sample_rag_chat_data["api_key"]
        )
        
        # Check that the request error was set correctly
        mock_request_tracker.set_request_error.assert_called_once()
        assert "Test error" in mock_request_tracker.set_request_error.call_args[0][1]
