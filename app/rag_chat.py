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
# Database Connection
# --------------------------------------------------------------------------- #

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

            # Use the PostgreSQL function to get diarized transcript instead of regular transcript
            sql = f"SELECT get_transcript_by_call_id({numeric_id}) as diarized_transcript;"
            query_result = await query(sql)

            if query_result and len(query_result) > 0:
                transcript_data = query_result[0].get('diarized_transcript')

                if transcript_data:
                    try:
                        # Parse the transcript data
                        # Handle the "TRANSCRIPT: " prefix if it exists
                        if isinstance(transcript_data, str):
                            if transcript_data.startswith("TRANSCRIPT: "):
                                json_str = transcript_data[len("TRANSCRIPT: "):]
                            else:
                                json_str = transcript_data
                            
                            # Parse the JSON
                            parsed_transcript = json.loads(json_str)
                        else:
                            # If it's already a dict/list (depending on how your DB returns it)
                            parsed_transcript = transcript_data

                        # Convert to the expected format with proper structure
                        formatted_transcript = []
                        for i, segment in enumerate(parsed_transcript):
                            formatted_segment = {
                                "speaker": segment.get("speaker", "UNKNOWN"),
                                "text": segment.get("text", ""),
                                "start": segment.get("start", i * 5.0),  # Default timing if not available
                                "end": segment.get("end", (i + 1) * 5.0)
                            }
                            formatted_transcript.append(formatted_segment)

                        # Create a properly formatted transcript object
                        results[call_id] = {
                            "id": call_id,
                            "date": "Current Date",  # You might want to get this from the DB too
                            "duration": "Unknown",   # You might want to get this from the DB too
                            "transcript": formatted_transcript
                        }
                        
                        logger.info(f"Successfully parsed transcript for call {call_id} with {len(formatted_transcript)} segments")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON for call {call_id}: {e}")
                        logger.error(f"Raw data: {transcript_data}")
                    except Exception as e:
                        logger.error(f"Error processing transcript for call {call_id}: {e}")

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
    if not transcript:
        logger.warning("Empty transcript provided to chunk_transcript")
        return []
    
    chunks = []

    for i in range(0, len(transcript), chunk_size):
        chunk_segments = transcript[i:i+chunk_size]

        # Create a formatted text from the segments with better readability
        chunk_lines = []
        for seg in chunk_segments:
            speaker = seg.get('speaker', 'UNKNOWN')
            text = seg.get('text', '')
            chunk_lines.append(f"{speaker}: {text}")
        
        chunk_text = "\n".join(chunk_lines)

        # Store chunk with metadata including individual segments for better attribution
        chunks.append({
            "text": chunk_text,
            "start_segment": i,
            "end_segment": min(i + chunk_size - 1, len(transcript) - 1),
            "segments": chunk_segments,
            "segment_count": len(chunk_segments)
        })

    logger.info(f"Created {len(chunks)} chunks from transcript with {len(transcript)} segments")
    return chunks

def format_source_attribution(chunk: Dict[str, Any], call_id: str, score: float) -> str:
    """
    Format a source attribution with specific segments for better readability.
    
    Args:
        chunk: The relevant chunk with segments
        call_id: The call ID
        score: Relevance score
        
    Returns:
        Formatted source attribution string
    """
    segments = chunk.get('segments', [])
    
    # Create a readable format showing the specific conversation parts
    attribution_lines = [
        f"**Source: Call {call_id} (Relevance: {score:.1%})**",
        ""
    ]
    
    for segment in segments:
        speaker = segment.get('speaker', 'UNKNOWN')
        text = segment.get('text', '')
        attribution_lines.append(f"**{speaker}:** {text}")
    
    return "\n".join(attribution_lines)

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
    # Create context from relevant chunks with better formatting
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(f"From Call {chunk['call_id']}:")
        # Format each segment properly
        for segment in chunk.get('segments', []):
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '')
            context_parts.append(f"{speaker}: {text}")
        context_parts.append("")  # Empty line between calls
    
    context = "\n".join(context_parts)

    # Create prompt for the model
    prompt = f"""
    You are an assistant that answers questions about customer service calls.
    Answer the following question based ONLY on the provided context from call transcripts.

    Context:
    {context}

    Question: {question}

    Provide a clear and concise answer. If the answer is not in the context, say "I don't have enough information to answer this question."
    Include specific references to which call(s) you got the information from and quote relevant parts of the conversation.
    """

    try:
        # Get model and generate response
        model = get_gemini_model(model_name, api_key)
        response = model.generate_content(prompt)
        answer = response.text.strip() if response.text else ""

        # Create source attributions with better formatting
        sources = []
        for chunk in relevant_chunks:
            if chunk["score"] > 0.3:  # Lower threshold to include more relevant sources
                # Format the source with specific conversation segments
                formatted_source = format_source_attribution(chunk, chunk["call_id"], chunk["score"])
                sources.append({
                    "call_id": chunk["call_id"],
                    "formatted_text": formatted_source,
                    "raw_segments": chunk.get('segments', []),
                    "score": chunk["score"],
                    "start_segment": chunk.get("start_segment", 0),
                    "end_segment": chunk.get("end_segment", 0)
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

# --------------------------------------------------------------------------- #
# Test Function
# --------------------------------------------------------------------------- #

async def test_transcript_parsing(call_id: str):
    """
    Test function to verify transcript parsing is working correctly.
    
    Args:
        call_id: The call ID to test with
    """
    print(f"Testing transcript parsing for call ID: {call_id}")
    
    try:
        # Test the database retrieval
        transcripts = await get_transcripts_from_db([call_id])
        
        if not transcripts:
            print("❌ No transcripts retrieved")
            return
            
        print(f"✅ Retrieved transcript data for {len(transcripts)} calls")
        
        # Test the chunking
        chunked_transcripts = process_transcripts_for_embeddings(transcripts)
        
        for call_id, chunks in chunked_transcripts.items():
            print(f"\n📞 Call {call_id}:")
            print(f"  - Number of chunks: {len(chunks)}")
            
            # Show first chunk as example
            if chunks:
                first_chunk = chunks[0]
                print(f"  - First chunk segments: {first_chunk['start_segment']} to {first_chunk['end_segment']}")
                print(f"  - First chunk text preview: {first_chunk['text'][:100]}...")
                
                # Show individual segments in first chunk
                print(f"  - Segments in first chunk:")
                for i, segment in enumerate(first_chunk['segments']):
                    speaker = segment.get('speaker', 'UNKNOWN')
                    text = segment.get('text', '')[:50]
                    print(f"    {i+1}. {speaker}: {text}...")
                    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

