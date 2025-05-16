"""
Test script for the RAG chat endpoint.
This script sends a test request to the /rag_chat endpoint and prints the response.
"""

import requests
import json
import sys

# URL of the API
BASE_URL = "http://localhost:8000"

def test_rag_chat():
    """Test the RAG chat endpoint with a sample question and call IDs."""
    
    # Sample request data
    data = {
        "question": "What was the customer's issue with their order?",
        "call_ids": ["call-123"]
    }
    
    # Send the request
    try:
        response = requests.post(f"{BASE_URL}/rag_chat", json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            
            print("\n=== RAG Chat Response ===")
            print(f"Question: {data['question']}")
            print("\nAnswer:")
            print(result["answer"])
            
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"\nSource {i} (Call {source['call_id']}, Score: {source['score']:.4f}):")
                print(source["text"])
                
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception occurred: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG chat endpoint...")
    success = test_rag_chat()
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed.")
        sys.exit(1)
