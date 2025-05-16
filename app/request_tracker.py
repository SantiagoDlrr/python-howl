"""
request_tracker.py
------------------------------------------------------------------
Module for tracking asynchronous requests in the application.
Provides functionality to create, update, and retrieve request status.
"""

import logging
import uuid
import time
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class RequestStatus(Enum):
    """Enum for request status values"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class RequestTracker:
    """
    Class for tracking asynchronous requests.
    Stores request status, results, and timestamps.
    """
    
    def __init__(self, cleanup_interval: int = 3600, max_age: int = 86400):
        """
        Initialize the request tracker.
        
        Args:
            cleanup_interval: Interval in seconds for cleaning up old requests (default: 1 hour)
            max_age: Maximum age in seconds for requests to be kept (default: 24 hours)
        """
        self.requests: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = cleanup_interval
        self.max_age = max_age
        self.last_cleanup = time.time()
    
    def create_request(self) -> str:
        """
        Create a new request and return its ID.
        
        Returns:
            str: The request ID
        """
        request_id = str(uuid.uuid4())
        self.requests[request_id] = {
            "status": RequestStatus.PENDING.value,
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None,
            "error": None
        }
        logger.info(f"Created new request with ID: {request_id}")
        return request_id
    
    def update_request_status(self, request_id: str, status: RequestStatus) -> bool:
        """
        Update the status of a request.
        
        Args:
            request_id: The ID of the request to update
            status: The new status
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        if request_id not in self.requests:
            logger.warning(f"Attempted to update non-existent request: {request_id}")
            return False
        
        self.requests[request_id]["status"] = status.value
        self.requests[request_id]["updated_at"] = time.time()
        logger.info(f"Updated request {request_id} status to {status.value}")
        return True
    
    def set_request_result(self, request_id: str, result: Any) -> bool:
        """
        Set the result of a request and mark it as completed.
        
        Args:
            request_id: The ID of the request
            result: The result data
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        if request_id not in self.requests:
            logger.warning(f"Attempted to set result for non-existent request: {request_id}")
            return False
        
        self.requests[request_id]["result"] = result
        self.requests[request_id]["status"] = RequestStatus.COMPLETED.value
        self.requests[request_id]["updated_at"] = time.time()
        logger.info(f"Set result for request {request_id} and marked as completed")
        return True
    
    def set_request_error(self, request_id: str, error: str) -> bool:
        """
        Set an error for a request and mark it as failed.
        
        Args:
            request_id: The ID of the request
            error: The error message
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        if request_id not in self.requests:
            logger.warning(f"Attempted to set error for non-existent request: {request_id}")
            return False
        
        self.requests[request_id]["error"] = error
        self.requests[request_id]["status"] = RequestStatus.FAILED.value
        self.requests[request_id]["updated_at"] = time.time()
        logger.info(f"Set error for request {request_id} and marked as failed: {error}")
        return True
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a request.
        
        Args:
            request_id: The ID of the request
            
        Returns:
            Optional[Dict[str, Any]]: The request status data, or None if not found
        """
        if request_id not in self.requests:
            logger.warning(f"Attempted to get status for non-existent request: {request_id}")
            return None
        
        request_data = self.requests[request_id]
        
        # Check if we need to clean up old requests
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests()
        
        return {
            "request_id": request_id,
            "status": request_data["status"],
            "created_at": request_data["created_at"],
            "updated_at": request_data["updated_at"],
            "result": request_data["result"],
            "error": request_data["error"]
        }
    
    def _cleanup_old_requests(self) -> None:
        """Clean up requests that are older than max_age"""
        current_time = time.time()
        to_remove = []
        
        for request_id, request_data in self.requests.items():
            if current_time - request_data["created_at"] > self.max_age:
                to_remove.append(request_id)
        
        for request_id in to_remove:
            del self.requests[request_id]
        
        self.last_cleanup = current_time
        logger.info(f"Cleaned up {len(to_remove)} old requests")

# Create a global instance of the request tracker
request_tracker = RequestTracker()
