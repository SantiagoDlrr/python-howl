"""
Unit tests for the request_tracker.py module.
"""
import time
import pytest
from unittest.mock import patch, MagicMock

from app.request_tracker import RequestTracker, RequestStatus

class TestRequestTracker:
    """Tests for the RequestTracker class."""

    def test_create_request(self):
        """Test creating a new request."""
        tracker = RequestTracker()
        request_id = tracker.create_request()
        
        # Check that the request was created
        assert request_id in tracker.requests
        assert tracker.requests[request_id]["status"] == RequestStatus.PENDING.value
        assert tracker.requests[request_id]["result"] is None
        assert tracker.requests[request_id]["error"] is None
        assert "created_at" in tracker.requests[request_id]
        assert "updated_at" in tracker.requests[request_id]
    
    def test_update_request_status(self):
        """Test updating the status of a request."""
        tracker = RequestTracker()
        request_id = tracker.create_request()
        
        # Update the status
        result = tracker.update_request_status(request_id, RequestStatus.PROCESSING)
        
        # Check the result and the updated status
        assert result is True
        assert tracker.requests[request_id]["status"] == RequestStatus.PROCESSING.value
    
    def test_update_request_status_nonexistent(self):
        """Test updating the status of a nonexistent request."""
        tracker = RequestTracker()
        
        # Try to update a nonexistent request
        result = tracker.update_request_status("nonexistent", RequestStatus.PROCESSING)
        
        # Check the result
        assert result is False
    
    def test_set_request_result(self):
        """Test setting the result of a request."""
        tracker = RequestTracker()
        request_id = tracker.create_request()
        
        # Set the result
        sample_result = {"answer": "Test answer", "sources": []}
        result = tracker.set_request_result(request_id, sample_result)
        
        # Check the result and the updated request
        assert result is True
        assert tracker.requests[request_id]["result"] == sample_result
        assert tracker.requests[request_id]["status"] == RequestStatus.COMPLETED.value
    
    def test_set_request_result_nonexistent(self):
        """Test setting the result of a nonexistent request."""
        tracker = RequestTracker()
        
        # Try to set the result of a nonexistent request
        sample_result = {"answer": "Test answer", "sources": []}
        result = tracker.set_request_result("nonexistent", sample_result)
        
        # Check the result
        assert result is False
    
    def test_set_request_error(self):
        """Test setting an error for a request."""
        tracker = RequestTracker()
        request_id = tracker.create_request()
        
        # Set the error
        error_message = "Test error"
        result = tracker.set_request_error(request_id, error_message)
        
        # Check the result and the updated request
        assert result is True
        assert tracker.requests[request_id]["error"] == error_message
        assert tracker.requests[request_id]["status"] == RequestStatus.FAILED.value
    
    def test_set_request_error_nonexistent(self):
        """Test setting an error for a nonexistent request."""
        tracker = RequestTracker()
        
        # Try to set an error for a nonexistent request
        error_message = "Test error"
        result = tracker.set_request_error("nonexistent", error_message)
        
        # Check the result
        assert result is False
    
    def test_get_request_status(self):
        """Test getting the status of a request."""
        tracker = RequestTracker()
        request_id = tracker.create_request()
        
        # Get the status
        status = tracker.get_request_status(request_id)
        
        # Check the status
        assert status["request_id"] == request_id
        assert status["status"] == RequestStatus.PENDING.value
        assert status["result"] is None
        assert status["error"] is None
        assert "created_at" in status
        assert "updated_at" in status
    
    def test_get_request_status_nonexistent(self):
        """Test getting the status of a nonexistent request."""
        tracker = RequestTracker()
        
        # Try to get the status of a nonexistent request
        status = tracker.get_request_status("nonexistent")
        
        # Check the result
        assert status is None
    
    @patch("time.time")
    def test_cleanup_old_requests(self, mock_time):
        """Test cleaning up old requests."""
        tracker = RequestTracker(cleanup_interval=10, max_age=100)
        
        # Create some requests with different timestamps
        mock_time.return_value = 1000
        request_id1 = tracker.create_request()
        
        mock_time.return_value = 1050
        request_id2 = tracker.create_request()
        
        mock_time.return_value = 1150
        
        # Trigger cleanup
        tracker._cleanup_old_requests()
        
        # Check that old requests were removed
        assert request_id1 not in tracker.requests
        assert request_id2 in tracker.requests
    
    @patch("time.time")
    def test_auto_cleanup_on_get_status(self, mock_time):
        """Test that cleanup is triggered when getting status after the cleanup interval."""
        tracker = RequestTracker(cleanup_interval=10, max_age=100)
        
        # Create a request
        mock_time.return_value = 1000
        request_id = tracker.create_request()
        tracker.last_cleanup = 1000
        
        # Get status before cleanup interval
        mock_time.return_value = 1005
        tracker.get_request_status(request_id)
        
        # Check that cleanup was not triggered
        assert tracker.last_cleanup == 1000
        
        # Get status after cleanup interval
        mock_time.return_value = 1015
        with patch.object(tracker, '_cleanup_old_requests') as mock_cleanup:
            tracker.get_request_status(request_id)
            mock_cleanup.assert_called_once()
