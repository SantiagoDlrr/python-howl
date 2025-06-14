# """ 
# Class to handle embeddings, insert and query
# """

# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone
# from dotenv import load_dotenv
# import os
# from embedding_query_result import EmbeddingQueryResult

# class Embeddings():

#     def __init__(self, model_name:str = 'intfloat/e5-small-v2') -> None:
#         """ Load the model and initialize Pinecone """
#         load_dotenv() 
#         pinecone_db = os.getenv("PINECONE_KEY")
#         self.embedding_model = SentenceTransformer(model_name)
#         self.pc = Pinecone(api_key=pinecone_db)
#         self.index = self.pc.Index("howl")
#         print("Model loaded")

#     def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
#         """ 
#         Generate embeddings for a list of texts 
#         Args:
#             texts (list): List of texts to generate embeddings for
#         Returns:
#             list: List of embeddings
#         """
#         return self.embedding_model.encode(texts)
    
#     def save_embedding(self, call_id, id: int, embedding: list[float], text: str) -> None:
#         """ 
#         Save the embedding to Pinecone 
#         Args:
#             call_id (str): The ID of the call
#             id (int): The ID of the embedding
#             embedding (list): The embedding to save
#             text (str): The text associated with the embedding
#         """
#         self.index.upsert([
#             (
#                 f"id-{call_id}-{id}",  # make ID globally unique
#                 embedding,
#                 {"text": text, "call_id": str(call_id)}
#             )
#         ]) 

#     def save_embeddings(self, call_id, embeddings: list[list[float]], texts: list[str]) -> None:
#         """ 
#         Save multiple embeddings to Pinecone 
#         Args:
#             call_id (str): The ID of the call
#             embeddings (list): List of embeddings to save
#             texts (list): List of texts associated with the embeddings
#         """
#         for i, (embedding, text) in enumerate(zip(embeddings, texts)):
#             self.save_embedding(
#                 call_id=call_id,
#                 id=i,
#                 embedding=embedding,
#                 text=text
#             )

#     def query(self, text, top_k=5):
#         """ 
#         Query the embedding 
#         Args:
#             text (str): The text to query
#             top_k (int): The number of top results to return
#         Returns:
#             results: The top_k results from the query
#         """
#         query_vector = self.embedding_model.encode([text])[0].tolist()  # embed the query
#         results = self.index.query(
#             vector=query_vector,
#             top_k=top_k,
#             include_metadata=True
#         )
#         json_results = self.parse_results(results.matches)
#         return json_results
    
#     def parse_results(self, results):
#         """ 
#         Parse the results from the query 
#         Args:
#             results: The results from the query
#         Returns:
#             list: List of parsed results
#         """
#         parsed_results = []
#         for result in results:
#             parsed_results.append(
#                 EmbeddingQueryResult(
#                     id=result.id,
#                     call_id=result.metadata["call_id"],
#                     text=result.metadata["text"],
#                     score=result.score
#                 )
#             )

#         json_results = [result.model_dump_json() for result in parsed_results]
#         return json_results

# if __name__ == "__main__":
#     """ Example usage """
#     emb = Embeddings()
#     chunks = [
#         "Agent: Hello, how can I help you?",
#         "Customer: My database went down after the update.",
#         "Agent: I see, let me check our logs...",
#     ]

#     # Save embeddings example:
#     # embeddings = emb.generate_embeddings(chunks)
#     # emb.save_embeddings(1, embeddings, chunks)

#     query = "Greeting"
#     result = emb.query(query, 1)
#     print(result)


"""
embeddings.py - Fixed version with better error handling and timeout management
"""

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time
import logging
from typing import List, Optional
from embedding_query_result import EmbeddingQueryResult

logger = logging.getLogger(__name__)

class Embeddings:
    def __init__(self, model_name: str = 'intfloat/e5-small-v2', max_retries: int = 3, timeout: int = 900) -> None:
        """
        Load the model and initialize Pinecone with better error handling
        
        Args:
            model_name: The sentence transformer model to use
            max_retries: Maximum number of download retries
            timeout: Timeout in seconds for model download
        """
        load_dotenv()
        
        # Initialize Pinecone first (faster operation)
        pinecone_key = os.getenv("PINECONE_KEY")
        if not pinecone_key:
            logger.error("PINECONE_KEY not found in environment variables")
            raise ValueError("PINECONE_KEY environment variable is required")
            
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index("howl")
        logger.info("Pinecone initialized successfully")
        
        # Load embedding model with retry logic
        self.embedding_model = self._load_model_with_retry(model_name, max_retries, timeout)
        logger.info(f"Embedding model '{model_name}' loaded successfully")

    def _load_model_with_retry(self, model_name: str, max_retries: int, timeout: int) -> SentenceTransformer:
        """
        Load SentenceTransformer model with retry logic and timeout handling
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading model '{model_name}' (attempt {attempt + 1}/{max_retries})")
                
                # Set environment variables for better download behavior
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(timeout)
                os.environ['TRANSFORMERS_CACHE'] = './model_cache'
                os.environ['HF_HOME'] = './model_cache'
                
                # Create cache directory
                os.makedirs('./model_cache', exist_ok=True)
                
                # Load model with explicit cache directory
                model = SentenceTransformer(
                    model_name, 
                    cache_folder='./model_cache',
                    device='cpu'  # Explicitly use CPU to avoid GPU issues
                )
                
                logger.info(f"Model '{model_name}' loaded successfully on attempt {attempt + 1}")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model on attempt {attempt + 1}: {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # Progressive backoff: 30s, 60s, 90s
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to load model after {max_retries} attempts")
                    # Try fallback to a smaller, more reliable model
                    return self._load_fallback_model()

    def _load_fallback_model(self) -> Optional[SentenceTransformer]:
        """
        Load a smaller, more reliable model as fallback
        """
        fallback_models = [
            'all-MiniLM-L6-v2',  # Smaller, faster model
            'paraphrase-MiniLM-L6-v2',  # Even smaller
        ]
        
        for fallback_model in fallback_models:
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                model = SentenceTransformer(
                    fallback_model,
                    cache_folder='./model_cache',
                    device='cpu'
                )
                logger.info(f"Fallback model '{fallback_model}' loaded successfully")
                return model
                
            except Exception as e:
                logger.error(f"Fallback model '{fallback_model}' also failed: {str(e)}")
                continue
        
        # If all models fail, raise an exception
        raise RuntimeError("Unable to load any embedding model. Check your internet connection and try again.")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with error handling
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings as lists of floats
        """
        if not texts:
            logger.warning("Empty text list provided to generate_embeddings")
            return []
            
        try:
            # Filter out empty or None texts
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                logger.warning("No valid texts found after filtering")
                return []
            
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            embeddings = self.embedding_model.encode(valid_texts, show_progress_bar=True)
            
            # Convert numpy arrays to lists
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            logger.info(f"Successfully generated {len(embeddings_list)} embeddings")
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

    def save_embedding(self, call_id: str, id: int, embedding: List[float], text: str) -> None:
        """
        Save a single embedding to Pinecone with error handling
        
        Args:
            call_id: The ID of the call
            id: The ID of the embedding
            embedding: The embedding vector
            text: The text associated with the embedding
        """
        try:
            vector_id = f"id-{call_id}-{id}"
            
            self.index.upsert([
                (
                    vector_id,
                    embedding,
                    {"text": text, "call_id": str(call_id)}
                )
            ])
            
            logger.debug(f"Successfully saved embedding {vector_id}")
            
        except Exception as e:
            logger.error(f"Failed to save embedding {vector_id}: {str(e)}")
            raise

    def save_embeddings(self, call_id: str, embeddings: List[List[float]], texts: List[str]) -> None:
        """
        Save multiple embeddings to Pinecone with batch processing
        
        Args:
            call_id: The ID of the call
            embeddings: List of embedding vectors
            texts: List of texts associated with the embeddings
        """
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")
            
        try:
            # Batch upsert for better performance
            batch_size = 100  # Pinecone recommended batch size
            vectors = []
            
            for i, (embedding, text) in enumerate(zip(embeddings, texts)):
                vector_id = f"id-{call_id}-{i}"
                vectors.append((
                    vector_id,
                    embedding,
                    {"text": text, "call_id": str(call_id)}
                ))
                
                # Upsert in batches
                if len(vectors) >= batch_size:
                    self.index.upsert(vectors)
                    vectors = []
                    logger.debug(f"Upserted batch for call {call_id}")
            
            # Upsert remaining vectors
            if vectors:
                self.index.upsert(vectors)
                
            logger.info(f"Successfully saved {len(embeddings)} embeddings for call {call_id}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings for call {call_id}: {str(e)}")
            raise

    def query(self, text: str, top_k: int = 5, call_id_filter: Optional[str] = None):
        """
        Query embeddings with improved error handling
        
        Args:
            text: The text to query
            top_k: Number of top results to return
            call_id_filter: Optional filter to search only specific call
            
        Returns:
            List of query results
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty query text provided")
                return []
            
            # Generate query embedding
            query_vector = self.embedding_model.encode([text.strip()])[0].tolist()
            
            # Build filter if call_id specified
            filter_dict = None
            if call_id_filter:
                filter_dict = {"call_id": {"$eq": str(call_id_filter)}}
            
            # Query Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Parse and return results
            parsed_results = self.parse_results(results.matches)
            logger.info(f"Query returned {len(parsed_results)} results")
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise RuntimeError(f"Embedding query failed: {str(e)}")

    def parse_results(self, results) -> List[str]:
        """
        Parse query results into structured format
        
        Args:
            results: Raw results from Pinecone query
            
        Returns:
            List of JSON-serialized EmbeddingQueryResult objects
        """
        try:
            parsed_results = []
            
            for result in results:
                if hasattr(result, 'metadata') and result.metadata:
                    parsed_result = EmbeddingQueryResult(
                        id=result.id,
                        call_id=result.metadata.get("call_id", "unknown"),
                        text=result.metadata.get("text", ""),
                        score=float(result.score) if hasattr(result, 'score') else 0.0
                    )
                    parsed_results.append(parsed_result)
            
            # Convert to JSON strings
            json_results = [result.model_dump_json() for result in parsed_results]
            return json_results
            
        except Exception as e:
            logger.error(f"Failed to parse results: {str(e)}")
            return []

    def health_check(self) -> dict:
        """
        Perform a health check on the embeddings system
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Test embedding generation
            test_text = "This is a test query for health check"
            test_embedding = self.embedding_model.encode([test_text])[0]
            
            # Test Pinecone connection
            index_stats = self.index.describe_index_stats()
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "embedding_dimension": len(test_embedding),
                "pinecone_connected": True,
                "index_stats": {
                    "total_vector_count": index_stats.total_vector_count,
                    "dimension": index_stats.dimension
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": hasattr(self, 'embedding_model'),
                "pinecone_connected": hasattr(self, 'index')
            }


# Create a singleton instance for the application
_embeddings_instance = None

def get_embeddings_instance() -> Embeddings:
    """
    Get or create a singleton Embeddings instance
    This helps avoid loading the model multiple times
    """
    global _embeddings_instance
    
    if _embeddings_instance is None:
        logger.info("Creating new Embeddings instance")
        _embeddings_instance = Embeddings()
    
    return _embeddings_instance


if __name__ == "__main__":
    """Example usage and testing"""
    import json
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create embeddings instance
        emb = Embeddings()
        
        # Test data
        chunks = [
            "Agent: Hello, how can I help you today?",
            "Customer: My database went down after the recent update.",
            "Agent: I see, let me check our system logs for any issues...",
            "Customer: This is very urgent, we need this fixed immediately.",
            "Agent: I understand the urgency. I'm escalating this to our technical team."
        ]
        
        print("Testing embedding generation...")
        embeddings = emb.generate_embeddings(chunks)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Save embeddings (uncomment to test)
        # print("Saving embeddings to Pinecone...")
        # emb.save_embeddings("test-call-123", embeddings, chunks)
        # print("Embeddings saved successfully")
        
        # Test query
        print("Testing query...")
        query = "database problem"
        results = emb.query(query, top_k=3)
        
        print(f"Query results for '{query}':")
        for result_json in results:
            result = json.loads(result_json)
            print(f"  Score: {result['score']:.3f} - {result['text'][:100]}...")
        
        # Health check
        print("\nHealth check:")
        health = emb.health_check()
        print(json.dumps(health, indent=2))
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

