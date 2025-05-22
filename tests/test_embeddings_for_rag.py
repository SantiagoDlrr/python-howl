"""
Unit tests for the embeddings module used in RAG chat.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.embeddings.embeddings import Embeddings
from app.embeddings.embedding_query_result import EmbeddingQueryResult

class TestEmbeddingsForRAG:
    """Tests for the Embeddings class used in RAG chat."""

    @patch("app.embeddings.embeddings.SentenceTransformer")
    @patch("app.embeddings.embeddings.Pinecone")
    @patch("app.embeddings.embeddings.load_dotenv")
    @patch("app.embeddings.embeddings.os.getenv")
    def test_embeddings_initialization(self, mock_getenv, mock_load_dotenv, 
                                      mock_pinecone, mock_sentence_transformer):
        """Test initializing the Embeddings class."""
        # Mock the dependencies
        mock_getenv.return_value = "fake-pinecone-key"
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        mock_pinecone_instance = MagicMock()
        mock_pinecone.return_value = mock_pinecone_instance
        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index
        
        # Initialize the Embeddings class
        embeddings = Embeddings()
        
        # Check that the dependencies were called correctly
        mock_load_dotenv.assert_called_once()
        mock_getenv.assert_called_once_with("PINECONE_KEY")
        mock_sentence_transformer.assert_called_once_with("intfloat/e5-large-v2")
        mock_pinecone.assert_called_once_with(api_key="fake-pinecone-key")
        mock_pinecone_instance.Index.assert_called_once_with("howl")
        
        # Check that the attributes were set correctly
        assert embeddings.embedding_model == mock_model
        assert embeddings.pc == mock_pinecone_instance
        assert embeddings.index == mock_index
    
    @patch("app.embeddings.embeddings.SentenceTransformer")
    @patch("app.embeddings.embeddings.Pinecone")
    @patch("app.embeddings.embeddings.load_dotenv")
    @patch("app.embeddings.embeddings.os.getenv")
    def test_generate_embeddings(self, mock_getenv, mock_load_dotenv, 
                               mock_pinecone, mock_sentence_transformer):
        """Test generating embeddings."""
        # Mock the dependencies
        mock_getenv.return_value = "fake-pinecone-key"
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize the Embeddings class
        embeddings = Embeddings()
        
        # Generate embeddings
        texts = ["Hello", "World"]
        result = embeddings.generate_embeddings(texts)
        
        # Check the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert np.array_equal(result, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
        mock_model.encode.assert_called_once_with(texts)
    
    @patch("app.embeddings.embeddings.SentenceTransformer")
    @patch("app.embeddings.embeddings.Pinecone")
    @patch("app.embeddings.embeddings.load_dotenv")
    @patch("app.embeddings.embeddings.os.getenv")
    def test_save_embedding(self, mock_getenv, mock_load_dotenv, 
                          mock_pinecone, mock_sentence_transformer):
        """Test saving an embedding."""
        # Mock the dependencies
        mock_getenv.return_value = "fake-pinecone-key"
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        mock_pinecone_instance = MagicMock()
        mock_pinecone.return_value = mock_pinecone_instance
        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index
        
        # Initialize the Embeddings class
        embeddings = Embeddings()
        
        # Save an embedding
        call_id = "call-123"
        id = 1
        embedding = [0.1, 0.2, 0.3]
        text = "Hello, world!"
        embeddings.save_embedding(call_id, id, embedding, text)
        
        # Check that the index.upsert method was called correctly
        mock_index.upsert.assert_called_once_with([
            (
                f"id-{call_id}-{id}",
                embedding,
                {"text": text, "call_id": str(call_id)}
            )
        ])
    
    @patch("app.embeddings.embeddings.SentenceTransformer")
    @patch("app.embeddings.embeddings.Pinecone")
    @patch("app.embeddings.embeddings.load_dotenv")
    @patch("app.embeddings.embeddings.os.getenv")
    def test_save_embeddings(self, mock_getenv, mock_load_dotenv, 
                           mock_pinecone, mock_sentence_transformer):
        """Test saving multiple embeddings."""
        # Mock the dependencies
        mock_getenv.return_value = "fake-pinecone-key"
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        mock_pinecone_instance = MagicMock()
        mock_pinecone.return_value = mock_pinecone_instance
        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index
        
        # Initialize the Embeddings class
        embeddings = Embeddings()
        
        # Save multiple embeddings
        call_id = "call-123"
        embeddings_list = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        texts = ["Hello", "World"]
        
        with patch.object(embeddings, 'save_embedding') as mock_save_embedding:
            embeddings.save_embeddings(call_id, embeddings_list, texts)
            
            # Check that save_embedding was called for each embedding
            assert mock_save_embedding.call_count == 2
            mock_save_embedding.assert_any_call(call_id=call_id, id=0, embedding=embeddings_list[0], text=texts[0])
            mock_save_embedding.assert_any_call(call_id=call_id, id=1, embedding=embeddings_list[1], text=texts[1])
    
    @patch("app.embeddings.embeddings.SentenceTransformer")
    @patch("app.embeddings.embeddings.Pinecone")
    @patch("app.embeddings.embeddings.load_dotenv")
    @patch("app.embeddings.embeddings.os.getenv")
    def test_query(self, mock_getenv, mock_load_dotenv, 
                 mock_pinecone, mock_sentence_transformer):
        """Test querying embeddings."""
        # Mock the dependencies
        mock_getenv.return_value = "fake-pinecone-key"
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_model
        mock_pinecone_instance = MagicMock()
        mock_pinecone.return_value = mock_pinecone_instance
        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index
        
        # Mock the query result
        mock_match1 = MagicMock()
        mock_match1.id = "id-call-123-0"
        mock_match1.metadata = {"call_id": "call-123", "text": "Hello"}
        mock_match1.score = 0.9
        
        mock_match2 = MagicMock()
        mock_match2.id = "id-call-456-0"
        mock_match2.metadata = {"call_id": "call-456", "text": "World"}
        mock_match2.score = 0.8
        
        mock_query_result = MagicMock()
        mock_query_result.matches = [mock_match1, mock_match2]
        mock_index.query.return_value = mock_query_result
        
        # Initialize the Embeddings class
        embeddings = Embeddings()
        
        # Mock the parse_results method
        with patch.object(embeddings, 'parse_results') as mock_parse_results:
            mock_parse_results.return_value = ["result1", "result2"]
            
            # Query embeddings
            result = embeddings.query("Hello", top_k=2)
            
            # Check the result
            assert result == ["result1", "result2"]
            mock_model.encode.assert_called_once_with(["Hello"])
            mock_index.query.assert_called_once_with(
                vector=[0.1, 0.2, 0.3],
                top_k=2,
                include_metadata=True
            )
            mock_parse_results.assert_called_once_with(mock_query_result.matches)
