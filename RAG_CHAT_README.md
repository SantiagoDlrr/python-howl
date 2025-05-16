# RAG Chat Implementation

This implementation adds a new RAG (Retrieval-Augmented Generation) chat endpoint to the existing application. The RAG chat endpoint allows users to ask questions about call transcripts and receive answers with source attribution.

## Features

- **Semantic Search**: Uses embeddings to find the most relevant parts of call transcripts
- **Source Attribution**: Includes references to the original calls in the response
- **Simple UI**: Includes a basic HTML interface for testing

## How to Use

### Starting the Server

1. Make sure you have all the required dependencies installed:
   ```
   pip install -r app/requirements.txt
   ```

2. Set the required environment variables:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. Start the FastAPI server:
   ```
   cd app
   python main.py
   ```

### Using the Web Interface

1. Open your browser and navigate to:
   ```
   http://localhost:8000/rag
   ```

2. Enter your question in the text area
3. Select the call IDs you want to search
4. Click "Submit Question"
5. View the answer and source attributions in the response panel

### Using the API Directly

You can also call the API endpoint directly:

```python
import requests

url = "http://localhost:8000/rag_chat"
data = {
    "question": "What was the customer's issue with their order?",
    "call_ids": ["call-123", "call-456"]
}

response = requests.post(url, json=data)
result = response.json()

print(result["answer"])
for source in result["sources"]:
    print(f"Source from call {source['call_id']}: {source['text']}")
```

## Implementation Details

### Files Added/Modified

- **app/rag_chat.py**: Contains the RAG chat implementation
- **app/main.py**: Updated to include the new endpoint
- **static/rag_chat.html**: Simple HTML interface for testing
- **app/requirements.txt**: Updated with new dependencies

### Fake Database

For testing purposes, the implementation includes a fake database with sample call transcripts. In a production environment, you would replace this with a real database connection.

Sample call IDs available for testing:
- `call-123`: Order delivery issue
- `call-456`: Printer WiFi setup

## Next Steps

1. Replace the fake database with a real PostgreSQL connection
2. Implement more sophisticated chunking strategies
3. Add caching for embeddings to improve performance
4. Implement more advanced RAG techniques
5. Enhance the UI with more features
