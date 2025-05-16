"""
rag_chat.py
------------------------------------------------------------------
RAG-based chat module for answering questions about call transcripts:
  • Retrieves transcripts from database using call IDs
  • Performs semantic search over transcript chunks
  • Uses RAG to generate answers with source attribution
  • Supports both synchronous and asynchronous processing
"""

import json
import logging
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# For embeddings and semantic search
from embeddings.embeddings import Embeddings
from embeddings.embedding_query_result import EmbeddingQueryResult

# For LLM
import google.generativeai as genai
from functools import lru_cache

# For request tracking
from request_tracker import request_tracker, RequestStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add near the top of the file, after imports
from huggingface_hub import login

# Initialize Hugging Face with token
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    logger.info("Logged in to Hugging Face Hub")
else:
    logger.warning("HF_TOKEN not set - model downloads may fail")

# --------------------------------------------------------------------------- #
# Fake Database Connection (for testing without actual PostgreSQL)
# --------------------------------------------------------------------------- #

# Sample transcript data (mimicking what would come from a database)
SAMPLE_TRANSCRIPTS = {
    "call-123": {
        "id": "call-123",
        "date": "15/05/2023",
        "duration": "00:05:23",
        "transcript": [
            {"speaker": "AGENT", "text": "Hello, thank you for calling customer support. How can I help you today?", "start": 0.0, "end": 4.5},
            {"speaker": "CUSTOMER", "text": "Hi, I'm having trouble with my recent order. It was supposed to arrive yesterday but I haven't received it yet.", "start": 4.8, "end": 10.2},
            {"speaker": "AGENT", "text": "I'm sorry to hear that. Let me check the status of your order. Can you provide your order number please?", "start": 10.5, "end": 15.0},
            {"speaker": "CUSTOMER", "text": "Yes, it's ABC12345.", "start": 15.3, "end": 17.0},
            {"speaker": "AGENT", "text": "Thank you. I can see your order was shipped on Monday, but there seems to be a delay with the courier. According to the tracking information, it should be delivered by end of day today.", "start": 17.5, "end": 25.0},
            {"speaker": "CUSTOMER", "text": "That's a relief. I was worried it got lost.", "start": 25.3, "end": 28.0},
            {"speaker": "AGENT", "text": "I understand your concern. If you don't receive it by tomorrow, please call us back and we'll file a claim with the shipping company.", "start": 28.5, "end": 35.0},
            {"speaker": "CUSTOMER", "text": "Okay, thank you for checking. I appreciate your help.", "start": 35.5, "end": 38.0},
            {"speaker": "AGENT", "text": "You're welcome. Is there anything else I can assist you with today?", "start": 38.5, "end": 41.0},
            {"speaker": "CUSTOMER", "text": "No, that's all. Thank you.", "start": 41.5, "end": 43.0},
            {"speaker": "AGENT", "text": "Thank you for calling. Have a great day!", "start": 43.5, "end": 46.0}
        ]
    },
    "call-456": {
        "id": "call-456",
        "date": "20/05/2023",
        "duration": "00:08:15",
        "transcript": [
            {"speaker": "AGENT", "text": "Thank you for calling technical support. My name is Alex. How may I assist you today?", "start": 0.0, "end": 5.0},
            {"speaker": "CUSTOMER", "text": "Hi Alex, I'm having issues connecting my new printer to my WiFi network.", "start": 5.5, "end": 10.0},
            {"speaker": "AGENT", "text": "I'd be happy to help you with that. What model of printer do you have?", "start": 10.5, "end": 14.0},
            {"speaker": "CUSTOMER", "text": "It's the HP OfficeJet Pro 9015.", "start": 14.5, "end": 16.0},
            {"speaker": "AGENT", "text": "Great. First, let's make sure your printer is in setup mode. There should be a blinking blue light on the control panel.", "start": 16.5, "end": 23.0},
            {"speaker": "CUSTOMER", "text": "Yes, I see the blinking light.", "start": 23.5, "end": 25.0},
            {"speaker": "AGENT", "text": "Perfect. Now on your computer or smartphone, you'll need to download the HP Smart app if you haven't already.", "start": 25.5, "end": 32.0},
            {"speaker": "CUSTOMER", "text": "I have the app installed already.", "start": 32.5, "end": 34.0},
            {"speaker": "AGENT", "text": "Excellent. Open the app and click on 'Add Printer'. The app should detect your printer in setup mode.", "start": 34.5, "end": 41.0},
            {"speaker": "CUSTOMER", "text": "Okay, it found my printer. Now it's asking for my WiFi password.", "start": 41.5, "end": 45.0},
            {"speaker": "AGENT", "text": "Go ahead and enter your WiFi password. Make sure it's entered correctly as passwords are case-sensitive.", "start": 45.5, "end": 51.0},
            {"speaker": "CUSTOMER", "text": "Done. It says it's connecting now... Oh, it worked! The printer is now connected to my WiFi.", "start": 51.5, "end": 58.0},
            {"speaker": "AGENT", "text": "That's great news! Let's make sure it's working properly. Can you try printing a test page?", "start": 58.5, "end": 63.0},
            {"speaker": "CUSTOMER", "text": "Yes, the test page printed perfectly. Thank you so much for your help!", "start": 63.5, "end": 68.0},
            {"speaker": "AGENT", "text": "You're welcome. Is there anything else I can help you with today?", "start": 68.5, "end": 71.0},
            {"speaker": "CUSTOMER", "text": "No, that's all I needed. Thanks again.", "start": 71.5, "end": 73.0},
            {"speaker": "AGENT", "text": "Thank you for calling technical support. Have a wonderful day!", "start": 73.5, "end": 77.0}
        ]
    }
}

async def get_transcripts_from_db(call_ids: List[str]) -> Dict[str, Any]:
    """
    Retrieve transcripts from PostgreSQL database using call IDs.

    Args:
        call_ids: List of call IDs to retrieve

    Returns:
        Dictionary mapping call_ids to their transcript data
    """
    logger.info(f"Retrieving transcripts for call IDs: {call_ids}")

    results = {}

    try:
        from db import query

        for call_id in call_ids:
            # Extract numeric ID from call-id format if needed
            numeric_id = call_id
            if call_id.startswith("call-"):
                numeric_id = call_id.split("-")[1]

            # Use the PostgreSQL function to get transcript
            sql = f"SELECT get_transcript_by_call_id({numeric_id}) as transcript;"
            query_result = await query(sql)

            if query_result and len(query_result) > 0:
                transcript_text = query_result[0].get('transcript')

                if transcript_text:
                    # Create a transcript object with the retrieved text
                    results[call_id] = {
                        "id": call_id,
                        "date": "Current Date",
                        "duration": "Unknown",
                        "transcript": [
                            {"speaker": "TRANSCRIPT", "text": transcript_text, "start": 0.0, "end": 0.0}
                        ]
                    }
    except Exception as e:
        logger.error(f"Database query failed: {e}")

    return results

# --------------------------------------------------------------------------- #
# Transcript Processing
# --------------------------------------------------------------------------- #

def chunk_transcript(transcript: List[Dict[str, Any]], chunk_size: int = 3) -> List[Dict[str, Any]]:
    """
    Split a transcript into chunks of specified size.

    Args:
        transcript: List of transcript segments
        chunk_size: Number of segments per chunk

    Returns:
        List of chunks, each containing segment data and metadata
    """
    chunks = []

    for i in range(0, len(transcript), chunk_size):
        chunk_segments = transcript[i:i+chunk_size]

        # Create a single text from the segments
        chunk_text = " ".join([f"{seg['speaker']}: {seg['text']}" for seg in chunk_segments])

        # Store chunk with metadata
        chunks.append({
            "text": chunk_text,
            "start_segment": i,
            "end_segment": min(i + chunk_size - 1, len(transcript) - 1),
            "segments": chunk_segments
        })

    return chunks

# --------------------------------------------------------------------------- #
# Embeddings and Semantic Search
# --------------------------------------------------------------------------- #

def initialize_embeddings_model() -> Embeddings:
    """Initialize and return the embeddings model"""
    try:
        return Embeddings()
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {e}")
        raise

def process_transcripts_for_embeddings(transcripts: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process transcripts into chunks suitable for embedding.

    Args:
        transcripts: Dictionary of transcripts by call_id

    Returns:
        Dictionary mapping call_ids to their chunked transcripts
    """
    chunked_transcripts = {}

    for call_id, transcript_data in transcripts.items():
        chunks = chunk_transcript(transcript_data["transcript"])
        chunked_transcripts[call_id] = chunks

    return chunked_transcripts

def perform_semantic_search(question: str, chunked_transcripts: Dict[str, List[Dict[str, Any]]],
                           embeddings_model: Embeddings, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic search over transcript chunks.

    Args:
        question: User's question
        chunked_transcripts: Dictionary of chunked transcripts by call_id
        embeddings_model: Initialized embeddings model
        top_k: Number of top results to return

    Returns:
        List of relevant chunks with their metadata
    """
    # In a real implementation, we would use the embeddings from the database
    # For this demo, we'll generate embeddings on the fly

    # Flatten all chunks from all transcripts
    all_chunks = []
    chunk_mapping = {}  # To map back to original call_id and chunk

    for call_id, chunks in chunked_transcripts.items():
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk["text"])
            chunk_mapping[len(all_chunks) - 1] = {"call_id": call_id, "chunk_index": i}

    # Generate embeddings for all chunks
    chunk_embeddings = embeddings_model.generate_embeddings(all_chunks)

    # Generate embedding for the question
    question_embedding = embeddings_model.generate_embeddings([question])[0]

    # Calculate similarity scores (simplified for demo)
    # In a real implementation, we would use a vector database like Pinecone
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]

    # Get top-k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        call_id = chunk_mapping[idx]["call_id"]
        chunk_index = chunk_mapping[idx]["chunk_index"]
        chunk = chunked_transcripts[call_id][chunk_index]

        results.append({
            "call_id": call_id,
            "text": chunk["text"],
            "score": float(similarities[idx]),
            "segments": chunk["segments"],
            "start_segment": chunk["start_segment"],
            "end_segment": chunk["end_segment"]
        })

    return results

# --------------------------------------------------------------------------- #
# RAG Pipeline
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=8)
def get_gemini_model(model_name: str, api_key: str):
    """Get or create a Gemini model instance"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def generate_rag_response(question: str, relevant_chunks: List[Dict[str, Any]],
                         model_name: str, api_key: str) -> Dict[str, Any]:
    """
    Generate a response using RAG with the Gemini model.

    Args:
        question: User's question
        relevant_chunks: List of relevant transcript chunks
        model_name: Name of the Gemini model to use
        api_key: Gemini API key

    Returns:
        Dictionary containing the answer and source attributions
    """
    # Create context from relevant chunks
    context = "\n\n".join([
        f"From Call {chunk['call_id']}:\n{chunk['text']}"
        for chunk in relevant_chunks
    ])

    # Create prompt for the model
    prompt = f"""
    You are an assistant that answers questions about customer service calls.
    Answer the following question based ONLY on the provided context from call transcripts.

    Context:
    {context}

    Question: {question}

    Provide a clear and concise answer. If the answer is not in the context, say "I don't have enough information to answer this question."
    Include specific references to which call(s) you got the information from.
    """

    try:
        # Get model and generate response
        model = get_gemini_model(model_name, api_key)
        response = model.generate_content(prompt)
        answer = response.text.strip() if response.text else ""

        # Create source attributions
        sources = []
        for chunk in relevant_chunks:
            if chunk["score"] > 0.5:  # Only include relevant sources
                sources.append({
                    "call_id": chunk["call_id"],
                    "text": chunk["text"],
                    "score": chunk["score"]
                })

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return {
            "answer": "Sorry, I encountered an error while processing your question.",
            "sources": []
        }

# --------------------------------------------------------------------------- #
# Main RAG Chat Function
# --------------------------------------------------------------------------- #

async def rag_chat(question: str, call_ids: List[str], model_name: str, api_key: str) -> Dict[str, Any]:
    """
    Main function to handle RAG-based chat (synchronous version).

    Args:
        question: User's question
        call_ids: List of call IDs to search in
        model_name: Name of the Gemini model to use
        api_key: Gemini API key

    Returns:
        Dictionary containing the answer and source attributions
    """
    try:
        # 1. Retrieve transcripts from database (now async)
        transcripts = await get_transcripts_from_db(call_ids)

        if not transcripts:
            return {
                "answer": "No transcripts found for the provided call IDs.",
                "sources": []
            }

        # 2. Process transcripts into chunks
        chunked_transcripts = process_transcripts_for_embeddings(transcripts)

        # 3. Initialize embeddings model
        embeddings_model = initialize_embeddings_model()

        # 4. Perform semantic search
        relevant_chunks = perform_semantic_search(question, chunked_transcripts, embeddings_model)

        # 5. Generate RAG response
        response = generate_rag_response(question, relevant_chunks, model_name, api_key)

        return response
    except Exception as e:
        logger.error(f"RAG chat error: {e}")
        return {
            "answer": f"An error occurred while processing your question: {str(e)}",
            "sources": []
        }

async def process_rag_chat_request(request_id: str, question: str, call_ids: List[str], model_name: str, api_key: str) -> None:
    """
    Process a RAG chat request asynchronously.
    This function is designed to be used with FastAPI's BackgroundTasks.

    Args:
        request_id: The ID of the request
        question: User's question
        call_ids: List of call IDs to search in
        model_name: Name of the Gemini model to use
        api_key: Gemini API key
    """
    logger.info(f"Starting asynchronous RAG chat for request {request_id}")

    # Update request status to processing
    request_tracker.update_request_status(request_id, RequestStatus.PROCESSING)

    try:
        # Call the RAG chat function
        response = await rag_chat(question, call_ids, model_name, api_key)

        # Update request with the result
        request_tracker.set_request_result(request_id, response)
        logger.info(f"Completed asynchronous RAG chat for request {request_id}")

    except Exception as e:
        # Update request with the error
        error_message = f"Error processing RAG chat request: {str(e)}"
        logger.error(error_message)
        request_tracker.set_request_error(request_id, error_message)

def create_rag_chat_request() -> str:
    """
    Create a new RAG chat request and return the request ID.

    Returns:
        str: The request ID
    """
    # Create a new request
    request_id = request_tracker.create_request()
    return request_id
