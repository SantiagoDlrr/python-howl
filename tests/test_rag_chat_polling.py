"""
Test script for the polling-based RAG chat endpoints.
This script sends test requests to the /rag_chat/start and /rag_chat/status endpoints.
"""

import requests
import json
import time
import sys

# URL of the API
BASE_URL = "http://localhost:8000"

def test_rag_chat_polling():
    """Test the polling-based RAG chat endpoints with a sample question and call IDs."""
    
    # Sample request data
    data = {
        "question": "What was the customer's issue with their order?",
        "call_ids": ["call-123"]
    }
    
    print("\n=== Testing Polling-Based RAG Chat ===")
    print(f"Question: {data['question']}")
    print(f"Call IDs: {data['call_ids']}")
    
    # Step 1: Start a new RAG chat request
    try:
        print("\nStarting RAG chat request...")
        response = requests.post(f"{BASE_URL}/rag_chat/start", json=data)
        
        if response.status_code != 200:
            print(f"Error starting request: {response.status_code} - {response.text}")
            return False
        
        start_data = response.json()
        request_id = start_data["request_id"]
        initial_status = start_data["status"]
        
        print(f"Request started with ID: {request_id}")
        print(f"Initial status: {initial_status}")
        
    except Exception as e:
        print(f"Exception occurred while starting request: {e}")
        return False
    
    # Step 2: Poll for status updates
    try:
        print("\nPolling for status updates...")
        max_polls = 30  # Maximum number of polling attempts
        poll_interval = 1  # Seconds between polls
        
        for i in range(max_polls):
            print(f"Poll {i+1}/{max_polls}...", end=" ")
            
            response = requests.get(f"{BASE_URL}/rag_chat/status/{request_id}")
            
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                return False
            
            status_data = response.json()
            current_status = status_data["status"]
            
            print(f"Status: {current_status}")
            
            # If the request is completed or failed, break the polling loop
            if current_status in ["completed", "failed"]:
                break
            
            # Wait before the next poll
            time.sleep(poll_interval)
        
        # Check if we reached the maximum number of polls
        if i == max_polls - 1 and current_status not in ["completed", "failed"]:
            print("Maximum polling attempts reached without completion.")
            return False
        
        # Step 3: Display the final result
        if current_status == "completed" and "result" in status_data:
            result = status_data["result"]
            
            print("\n=== RAG Chat Response ===")
            print("\nAnswer:")
            print(result["answer"])
            
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"\nSource {i} (Call {source['call_id']}, Score: {source['score']:.4f}):")
                print(source["text"])
            
            return True
        elif current_status == "failed":
            print(f"\nRequest failed: {status_data.get('error', 'Unknown error')}")
            return False
        else:
            print("\nUnexpected status or missing result.")
            return False
            
    except Exception as e:
        print(f"Exception occurred while polling: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_chat_polling()
    sys.exit(0 if success else 1)
