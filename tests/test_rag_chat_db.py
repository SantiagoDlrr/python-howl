"""
Unit tests for the database interaction in the rag_chat.py module.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.rag_chat import get_transcripts_from_db, SAMPLE_TRANSCRIPTS

class TestRagChatDB:
    """Tests for the database interaction in the rag_chat.py module."""

    @pytest.mark.asyncio
    @patch("app.rag_chat.query")
    async def test_get_transcripts_from_db_success(self, mock_query):
        """Test retrieving transcripts from the database successfully."""
        # Mock the query function
        mock_query.return_value = [
            {"transcript": "This is a test transcript."}
        ]
        
        # Call the function
        result = await get_transcripts_from_db(["call-123"])
        
        # Check the result
        assert "call-123" in result
        assert result["call-123"]["id"] == "call-123"
        assert "transcript" in result["call-123"]
        assert len(result["call-123"]["transcript"]) == 1
        assert result["call-123"]["transcript"][0]["text"] == "This is a test transcript."
        
        # Check that the query function was called with the correct SQL
        mock_query.assert_called_once()
        assert "SELECT get_transcript_by_call_id" in mock_query.call_args[0][0]
        assert "123" in mock_query.call_args[0][0]
    
    @pytest.mark.asyncio
    @patch("app.rag_chat.query")
    async def test_get_transcripts_from_db_empty_result(self, mock_query):
        """Test retrieving transcripts from the database with an empty result."""
        # Mock the query function
        mock_query.return_value = []
        
        # Call the function
        result = await get_transcripts_from_db(["call-123"])
        
        # Check the result
        assert result == {}
        
        # Check that the query function was called with the correct SQL
        mock_query.assert_called_once()
        assert "SELECT get_transcript_by_call_id" in mock_query.call_args[0][0]
        assert "123" in mock_query.call_args[0][0]
    
    @pytest.mark.asyncio
    @patch("app.rag_chat.query")
    async def test_get_transcripts_from_db_null_transcript(self, mock_query):
        """Test retrieving transcripts from the database with a null transcript."""
        # Mock the query function
        mock_query.return_value = [
            {"transcript": None}
        ]
        
        # Call the function
        result = await get_transcripts_from_db(["call-123"])
        
        # Check the result
        assert result == {}
        
        # Check that the query function was called with the correct SQL
        mock_query.assert_called_once()
        assert "SELECT get_transcript_by_call_id" in mock_query.call_args[0][0]
        assert "123" in mock_query.call_args[0][0]
    
    @pytest.mark.asyncio
    @patch("app.rag_chat.query")
    async def test_get_transcripts_from_db_exception(self, mock_query):
        """Test retrieving transcripts from the database with an exception."""
        # Mock the query function to raise an exception
        mock_query.side_effect = Exception("Test error")
        
        # Call the function
        result = await get_transcripts_from_db(["call-123"])
        
        # Check the result
        assert result == {}
        
        # Check that the query function was called with the correct SQL
        mock_query.assert_called_once()
        assert "SELECT get_transcript_by_call_id" in mock_query.call_args[0][0]
        assert "123" in mock_query.call_args[0][0]
    
    @pytest.mark.asyncio
    @patch("app.rag_chat.query")
    async def test_get_transcripts_from_db_multiple_call_ids(self, mock_query):
        """Test retrieving transcripts from the database with multiple call IDs."""
        # Mock the query function
        mock_query.side_effect = [
            [{"transcript": "Transcript for call 123."}],
            [{"transcript": "Transcript for call 456."}]
        ]
        
        # Call the function
        result = await get_transcripts_from_db(["call-123", "call-456"])
        
        # Check the result
        assert "call-123" in result
        assert "call-456" in result
        assert result["call-123"]["transcript"][0]["text"] == "Transcript for call 123."
        assert result["call-456"]["transcript"][0]["text"] == "Transcript for call 456."
        
        # Check that the query function was called with the correct SQL
        assert mock_query.call_count == 2
        assert "123" in mock_query.call_args_list[0][0][0]
        assert "456" in mock_query.call_args_list[1][0][0]
    
    @pytest.mark.asyncio
    async def test_get_transcripts_from_db_sample_data(self):
        """Test retrieving transcripts from the sample data."""
        # Call the function with a call ID that exists in the sample data
        result = await get_transcripts_from_db(["call-123"])
        
        # Check the result
        assert "call-123" in result
        assert result["call-123"] == SAMPLE_TRANSCRIPTS["call-123"]
        
        # Call the function with a call ID that doesn't exist in the sample data
        result = await get_transcripts_from_db(["call-999"])
        
        # Check the result
        assert result == {}
