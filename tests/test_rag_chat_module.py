"""
Unit tests for the rag_chat.py module.

This test file covers the core functionality of the RAG (Retrieval-Augmented Generation) chat system:
1. Transcript chunking and processing
2. Embeddings model initialization
3. Semantic search over transcript chunks
4. RAG response generation with the Gemini model
5. The main rag_chat function that orchestrates the entire process

These tests use mocks to isolate the components being tested and verify that they
interact correctly with each other.
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Import the functions to test
from app.rag_chat import (
    get_transcripts_from_db,
    chunk_transcript,
    process_transcripts_for_embeddings,
    perform_semantic_search,
    generate_rag_response,
    rag_chat,
    process_rag_chat_request,
    create_rag_chat_request,
    initialize_embeddings_model,
    SAMPLE_TRANSCRIPTS
)
from app.request_tracker import RequestStatus

# Sample data for tests
SAMPLE_QUESTION = "What was the customer's issue with their order?"
SAMPLE_CALL_IDS = ["call-123"]
SAMPLE_MODEL_NAME = "gemini-1.5-flash"
SAMPLE_API_KEY = "fake-api-key"
SAMPLE_REQUEST_ID = "test-request-id"

class TestRagChatModule:
    """Tests for the rag_chat.py module."""

    def test_chunk_transcript(self):
        """
        Test chunking a transcript into segments.

        This test verifies that the chunk_transcript function:
        1. Correctly divides a transcript into chunks of the specified size
        2. Properly sets the start_segment and end_segment indices
        3. Includes the correct segments in each chunk
        4. Creates the expected text representation of each chunk
        5. Works with both default (3) and custom chunk sizes
        """
        # Sample transcript data
        transcript = [
            {"speaker": "AGENT", "text": "Hello, how can I help you?", "start": 0.0, "end": 3.0},
            {"speaker": "CUSTOMER", "text": "I have an issue with my order.", "start": 3.5, "end": 6.0},
            {"speaker": "AGENT", "text": "I'm sorry to hear that.", "start": 6.5, "end": 8.0},
            {"speaker": "AGENT", "text": "Let me check your order.", "start": 8.5, "end": 10.0},
            {"speaker": "CUSTOMER", "text": "Thank you.", "start": 10.5, "end": 11.0}
        ]

        # Test with default chunk size (3)
        chunks = chunk_transcript(transcript)
        assert len(chunks) == 2
        assert chunks[0]["start_segment"] == 0
        assert chunks[0]["end_segment"] == 2
        assert len(chunks[0]["segments"]) == 3
        assert "AGENT: Hello, how can I help you?" in chunks[0]["text"]

        # Test with custom chunk size
        chunks = chunk_transcript(transcript, chunk_size=2)
        assert len(chunks) == 3
        assert chunks[1]["start_segment"] == 2
        assert chunks[1]["end_segment"] == 3

    def test_process_transcripts_for_embeddings(self):
        """
        Test processing transcripts for embeddings.

        This test verifies that the process_transcripts_for_embeddings function:
        1. Takes a dictionary of transcripts and processes each one
        2. Correctly chunks each transcript using the chunk_transcript function
        3. Returns a dictionary with the same keys (call_ids) as the input
        4. Produces chunks with the expected structure (text, segments, etc.)

        This is a critical function that prepares transcript data for semantic search.
        """
        # Use a subset of the sample transcripts
        transcripts = {
            "call-123": SAMPLE_TRANSCRIPTS["call-123"]
        }

        chunked_transcripts = process_transcripts_for_embeddings(transcripts)

        assert "call-123" in chunked_transcripts
        assert len(chunked_transcripts["call-123"]) > 0
        assert "text" in chunked_transcripts["call-123"][0]
        assert "segments" in chunked_transcripts["call-123"][0]

    @patch("app.rag_chat.Embeddings")
    def test_initialize_embeddings_model(self, mock_embeddings_class):
        """
        Test initializing the embeddings model.

        This test verifies that the initialize_embeddings_model function:
        1. Creates an instance of the Embeddings class
        2. Returns that instance for use in semantic search
        3. Handles the initialization process correctly

        The embeddings model is essential for semantic search functionality,
        allowing the system to find relevant transcript chunks based on the user's question.
        """
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance

        result = initialize_embeddings_model()

        assert result == mock_embeddings_instance
        mock_embeddings_class.assert_called_once()

    @patch("app.rag_chat.cosine_similarity")
    @patch("app.rag_chat.np.argsort")
    def test_perform_semantic_search(self, mock_argsort, mock_cosine_similarity):
        """
        Test performing semantic search over transcript chunks.

        This test verifies that the perform_semantic_search function:
        1. Generates embeddings for the user's question
        2. Generates embeddings for all transcript chunks
        3. Calculates similarity scores between the question and chunks
        4. Identifies and returns the most relevant chunks
        5. Includes proper metadata (call_id, score, text, segments) in the results

        This is the core of the RAG system's retrieval component, finding the most
        relevant transcript chunks to use as context for generating an answer.
        """
        # Mock embeddings model
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.generate_embeddings.side_effect = [
            [[0.1, 0.2, 0.3]],  # Question embedding
            [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]  # Chunk embeddings
        ]

        # Mock numpy and sklearn functions
        mock_cosine_similarity.return_value = [[0.7, 0.9]]
        mock_argsort.return_value = [0, 1]

        # Sample chunked transcripts
        chunked_transcripts = {
            "call-123": [
                {
                    "text": "AGENT: Hello, how can I help you? CUSTOMER: I have an issue with my order.",
                    "start_segment": 0,
                    "end_segment": 1,
                    "segments": [
                        {"speaker": "AGENT", "text": "Hello, how can I help you?", "start": 0.0, "end": 3.0},
                        {"speaker": "CUSTOMER", "text": "I have an issue with my order.", "start": 3.5, "end": 6.0}
                    ]
                },
                {
                    "text": "AGENT: I'm sorry to hear that. AGENT: Let me check your order.",
                    "start_segment": 2,
                    "end_segment": 3,
                    "segments": [
                        {"speaker": "AGENT", "text": "I'm sorry to hear that.", "start": 6.5, "end": 8.0},
                        {"speaker": "AGENT", "text": "Let me check your order.", "start": 8.5, "end": 10.0}
                    ]
                }
            ]
        }

        results = perform_semantic_search(
            SAMPLE_QUESTION,
            chunked_transcripts,
            mock_embeddings_model,
            top_k=2
        )

        assert len(results) == 2
        assert results[0]["call_id"] == "call-123"
        assert "score" in results[0]
        assert "text" in results[0]
        assert "segments" in results[0]

    @patch("app.rag_chat.get_gemini_model")
    def test_generate_rag_response(self, mock_get_gemini_model):
        """
        Test generating a RAG response using the Gemini model.

        This test verifies that the generate_rag_response function:
        1. Creates a context from the relevant transcript chunks
        2. Constructs an appropriate prompt for the Gemini model
        3. Calls the Gemini model with the prompt
        4. Processes the model's response into the expected format
        5. Includes both the answer and source attributions in the response

        This function represents the "generation" part of RAG, using the retrieved
        context to generate an answer that's grounded in the transcript data.
        """
        # Mock Gemini model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "The customer had an issue with their order not arriving on time."
        mock_model.generate_content.return_value = mock_response
        mock_get_gemini_model.return_value = mock_model

        # Sample relevant chunks
        relevant_chunks = [
            {
                "call_id": "call-123",
                "text": "AGENT: Hello, how can I help you? CUSTOMER: I have an issue with my order.",
                "score": 0.9,
                "segments": []
            },
            {
                "call_id": "call-123",
                "text": "CUSTOMER: It was supposed to arrive yesterday but I haven't received it yet.",
                "score": 0.8,
                "segments": []
            }
        ]

        response = generate_rag_response(
            SAMPLE_QUESTION,
            relevant_chunks,
            SAMPLE_MODEL_NAME,
            SAMPLE_API_KEY
        )

        assert "answer" in response
        assert response["answer"] == "The customer had an issue with their order not arriving on time."
        assert "sources" in response
        assert len(response["sources"]) > 0

    @pytest.mark.asyncio
    @patch("app.rag_chat.get_transcripts_from_db")
    @patch("app.rag_chat.initialize_embeddings_model")
    @patch("app.rag_chat.perform_semantic_search")
    @patch("app.rag_chat.generate_rag_response")
    async def test_rag_chat(self, mock_generate_response, mock_perform_search,
                           mock_init_embeddings, mock_get_transcripts):
        """
        Test the main rag_chat function that orchestrates the entire RAG process.

        This test verifies that the rag_chat function:
        1. Retrieves transcripts from the database using the provided call IDs
        2. Processes the transcripts into chunks suitable for embedding
        3. Initializes the embeddings model
        4. Performs semantic search to find relevant chunks
        5. Generates a RAG response using the Gemini model
        6. Returns the final response with answer and sources

        This is the main entry point for the RAG chat functionality, coordinating
        all the components of the system to produce a final answer based on the
        transcript data.
        """
        # Mock the database query
        mock_get_transcripts.return_value = {"call-123": SAMPLE_TRANSCRIPTS["call-123"]}

        # Mock the embeddings model
        mock_embeddings_model = MagicMock()
        mock_init_embeddings.return_value = mock_embeddings_model

        # Mock semantic search
        mock_perform_search.return_value = [
            {
                "call_id": "call-123",
                "text": "CUSTOMER: I'm having trouble with my recent order. It was supposed to arrive yesterday but I haven't received it yet.",
                "score": 0.9,
                "segments": []
            }
        ]

        # Mock RAG response generation
        expected_response = {
            "answer": "The customer's order was delayed and hadn't arrived when expected.",
            "sources": [
                {
                    "call_id": "call-123",
                    "text": "CUSTOMER: I'm having trouble with my recent order. It was supposed to arrive yesterday but I haven't received it yet.",
                    "score": 0.9
                }
            ]
        }
        mock_generate_response.return_value = expected_response

        # Call the function
        response = await rag_chat(
            SAMPLE_QUESTION,
            SAMPLE_CALL_IDS,
            SAMPLE_MODEL_NAME,
            SAMPLE_API_KEY
        )

        # Verify the response
        assert response == expected_response
        mock_get_transcripts.assert_called_once_with(SAMPLE_CALL_IDS)
        mock_init_embeddings.assert_called_once()
        mock_perform_search.assert_called_once()
        mock_generate_response.assert_called_once()



# tests/test_rag_chat_endpoints.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app.main import app   # Ajusta si tu FastAPI vive en otro módulo

client = TestClient(app)


class TestRagChatEndpoint:
    """Pruebas del endpoint /rag_chat (AC1 y AC6)."""

    @pytest.mark.asyncio
    @patch("app.main.rag_chat_function", new_callable=AsyncMock)
    def test_rag_chat_success(self, mock_rag_chat):
        """AC1: siempre devuelve answer y sources."""
        mock_rag_chat.return_value = {
            "answer": "Llega hoy por la tarde.",
            "sources": [
                {"call_id": "call-123", "text": "…", "score": 0.92}
            ]
        }

        response = client.post(
            "/rag_chat",
            json={"question": "¿Cuándo llega el pedido?", "call_ids": ["call-123"]}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert isinstance(body["sources"], list) and len(body["sources"]) > 0
        mock_rag_chat.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("app.main.rag_chat_function", new_callable=AsyncMock)
    def test_rag_chat_service_error(self, mock_rag_chat):
        """AC6: error interno => 503 más mensaje."""
        mock_rag_chat.side_effect = RuntimeError("boom")

        response = client.post(
            "/rag_chat",
            json={"question": "foo?", "call_ids": ["call-123"]}
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "RAG chat service error."
# tests/test_rag_chat_module.py  (mismo archivo grande, o uno nuevo)
@patch("app.rag_chat.cosine_similarity")
@patch("app.rag_chat.np.argsort")
def test_perform_semantic_search_ordering(mock_argsort, mock_cosine_similarity):
    """AC3: los resultados llegan ordenados por score descendente."""
    from app.rag_chat import perform_semantic_search

    #  Generamos similitudes [0.95, 0.60, 0.42] ya ordenadas
    mock_cosine_similarity.return_value = [[0.95, 0.60, 0.42]]
    mock_argsort.return_value = [0, 1, 2]

    # Mock embeddings
    embedder = MagicMock()
    embedder.generate_embeddings.side_effect = [
        [[0.1, 0.1, 0.1]],                       # pregunta
        [[0.9, 0.9, 0.9], [0.6, 0.6, 0.6], [0]]  # chunks
    ]

    chunked = {"call-1": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}

    results = perform_semantic_search("q", chunked, embedder, top_k=3)

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)   # descendente



# tests/test_rag_chat_module.py  (añade después de los otros)
@patch("app.rag_chat.get_gemini_model")
def test_generate_rag_response_no_context(mock_get_model):
    """AC4: sin contexto suficiente -> disclaimer y sources vacías."""
    from app.rag_chat import generate_rag_response

    # Simulamos Gemini explicando que no hay info
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "I don't have enough information to answer this question."
    mock_model.generate_content.return_value = mock_resp
    mock_get_model.return_value = mock_model

    response = generate_rag_response(
        question="¿Color favorito?",
        relevant_chunks=[],          # sin contexto
        model_name="gemini-1.5-flash",
        api_key="fake"
    )

    assert "enough information" in response["answer"].lower()
    assert response["sources"] == []



# tests/test_rag_chat_module.py
@patch("app.rag_chat.get_gemini_model")
def test_generate_rag_response_threshold(mock_get_model):
    """AC5: solo se incluyen fuentes con score > 0.5."""
    from app.rag_chat import generate_rag_response

    # Stub Gemini
    model = MagicMock()
    reply = MagicMock()
    reply.text = "Respuesta cualquiera."
    model.generate_content.return_value = reply
    mock_get_model.return_value = model

    chunks = [
        {"call_id": "call-123", "text": "Alta relevancia", "score": 0.9, "segments": []},
        {"call_id": "call-123", "text": "Media relevancia", "score": 0.6, "segments": []},
        {"call_id": "call-123", "text": "Baja relevancia",  "score": 0.3, "segments": []},  # debe quedar fuera
    ]

    res = generate_rag_response("pregunta", chunks, "gemini-1.5-flash", "fake")

    assert all(src["score"] > 0.5 for src in res["sources"])
    call_texts = [src["text"] for src in res["sources"]]
    assert "Baja relevancia" not in call_texts



