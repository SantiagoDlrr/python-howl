# RAG Chat Tests

This directory contains unit tests for the RAG chat functionality in the Howl Python application.

## What are Unit Tests?

Unit tests are automated tests that verify individual components (or "units") of code work as expected in isolation. They help ensure that each function or method behaves correctly according to its specifications. Benefits of unit tests include:

1. **Early Bug Detection**: Catch issues early in the development process
2. **Regression Prevention**: Ensure new changes don't break existing functionality
3. **Documentation**: Tests serve as executable documentation of how code should behave
4. **Design Improvement**: Writing tests often leads to better code design
5. **Confidence in Refactoring**: Make changes with confidence that functionality remains intact

In the context of the RAG chat functionality, unit tests verify that each component of the system (transcript processing, semantic search, response generation, etc.) works correctly on its own and interacts properly with other components.

## Test Files

1. **test_rag_chat_module.py**: Tests for the core functionality in rag_chat.py
   - `test_chunk_transcript`: Verifies that the transcript chunking function correctly divides transcript data into manageable segments with proper metadata. Tests both default and custom chunk sizes to ensure chunks are created with the right start/end segment indices and text content.
   - `test_process_transcripts_for_embeddings`: Ensures that transcripts are properly processed into chunks suitable for embedding, validating that the output contains the expected structure with text and segment information.
   - `test_initialize_embeddings_model`: Confirms that the embeddings model is correctly initialized and returns the expected instance.
   - `test_perform_semantic_search`: Tests the semantic search functionality that finds relevant transcript chunks based on a user question. Verifies that search results contain proper call IDs, scores, and text content.
   - `test_generate_rag_response`: Validates that the RAG response generation correctly uses the Gemini model to create answers based on relevant chunks, and that responses include both answers and source attributions.
   - `test_rag_chat`: Tests the main RAG chat function that orchestrates the entire process from retrieving transcripts to generating the final response. Ensures all components are called with correct parameters and the final response has the expected format.

2. **test_rag_chat_endpoints.py**: Tests for the RAG chat endpoints in main.py
   - `/rag_chat`
   - `/rag_chat/start`
   - `/rag_chat/status/{request_id}`

3. **test_request_tracker.py**: Tests for the request_tracker.py module
   - `RequestTracker.create_request`
   - `RequestTracker.update_request_status`
   - `RequestTracker.set_request_result`
   - `RequestTracker.set_request_error`
   - `RequestTracker.get_request_status`
   - `RequestTracker._cleanup_old_requests`

4. **test_embeddings_for_rag.py**: Tests for the embeddings module used in RAG chat
   - `Embeddings.__init__`
   - `Embeddings.generate_embeddings`
   - `Embeddings.save_embedding`
   - `Embeddings.save_embeddings`
   - `Embeddings.query`

5. **test_rag_chat_async.py**: Tests for the asynchronous processing functionality
   - `create_rag_chat_request`
   - `process_rag_chat_request`

6. **test_rag_chat_db.py**: Tests for the database interaction
   - `get_transcripts_from_db`

## Running the Tests

You can run the tests using pytest. Here are the commands:

1. To run all tests:
```
pytest tests/
```

2. To run a specific test file:
```
pytest tests/test_rag_chat_module.py
```

3. To run tests with coverage:
```
pytest tests/ --cov=app.rag_chat --cov=app.request_tracker --cov-report=term
```

4. To run tests with HTML coverage report:
```
pytest tests/ --cov=app.rag_chat --cov=app.request_tracker --cov-report=html
```

## Detailed Look at test_rag_chat_module.py

The `test_rag_chat_module.py` file contains tests for the core RAG functionality. Here's a detailed explanation of what each test does:

### 1. `test_chunk_transcript`
This test verifies that transcripts are properly divided into manageable chunks. It tests:
- Creating chunks with the default size (3 segments per chunk)
- Creating chunks with a custom size (2 segments per chunk)
- Proper assignment of start and end segment indices
- Correct inclusion of transcript segments in each chunk
- Proper text formatting that combines speaker and text information

### 2. `test_process_transcripts_for_embeddings`
This test ensures that entire transcripts are properly processed into chunks ready for embedding:
- Takes a dictionary of transcripts indexed by call_id
- Processes each transcript into chunks
- Returns a dictionary with the same structure but containing chunked data
- Ensures each chunk has the necessary fields (text, segments, etc.)

### 3. `test_initialize_embeddings_model`
This test verifies that the embeddings model is correctly initialized:
- Creates an instance of the Embeddings class
- Returns the instance for use in semantic search
- Properly handles any initialization parameters

### 4. `test_perform_semantic_search`
This test checks the semantic search functionality:
- Generates embeddings for the user's question
- Generates embeddings for all transcript chunks
- Calculates similarity scores between the question and chunks
- Identifies the most relevant chunks based on similarity
- Returns results with proper metadata (call_id, score, text, segments)

### 5. `test_generate_rag_response`
This test validates the RAG response generation:
- Creates a context from relevant transcript chunks
- Constructs an appropriate prompt for the Gemini model
- Calls the model with the prompt
- Processes the model's response
- Returns a structured response with answer and source attributions

### 6. `test_rag_chat`
This test verifies the main function that orchestrates the entire RAG process:
- Retrieves transcripts from the database
- Processes transcripts into chunks
- Initializes the embeddings model
- Performs semantic search
- Generates a RAG response
- Returns the final response with answer and sources

## Test Coverage

The tests cover the following aspects of the RAG chat functionality:

1. **Transcript Processing**: Tests for chunking transcripts and preparing them for embedding.
2. **Semantic Search**: Tests for performing semantic search over transcript chunks.
3. **RAG Response Generation**: Tests for generating responses using RAG with the Gemini model.
4. **Asynchronous Processing**: Tests for the polling-based system for the RAG chat endpoint.
5. **Database Interaction**: Tests for retrieving transcripts from the database.
6. **Request Tracking**: Tests for tracking the status of asynchronous requests.
7. **Embeddings**: Tests for generating and querying embeddings.
8. **API Endpoints**: Tests for the RAG chat endpoints in the FastAPI application.

## Mocks and Fixtures

The tests use the following mocks and fixtures:

1. **mock_env_vars**: Mocks environment variables needed for tests.
2. **mock_oci_config**: Mocks the OCI configuration.
3. **mock_oci_client**: Mocks the OCI Language client.
4. **mock_embeddings**: Mocks the Embeddings class.
5. **mock_db_query**: Mocks the database query function.
6. **mock_gemini_model**: Mocks the Gemini model.
7. **sample_rag_chat_data**: Provides sample data for RAG chat tests.

These mocks and fixtures are defined in the `conftest.py` file.
