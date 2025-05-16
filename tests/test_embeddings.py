"""
Test script for the embeddings module.
This script tests the basic functionality of the embeddings module.
"""

import sys
import os
import numpy as np

# Add the app directory to the path so we can import the embeddings module
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

try:
    from embeddings.embeddings import Embeddings
except ImportError:
    print("Error: Could not import Embeddings module. Make sure the path is correct.")
    sys.exit(1)

def test_embeddings():
    """Test the basic functionality of the embeddings module."""
    
    print("Initializing embeddings model...")
    try:
        # Initialize the embeddings model
        embeddings_model = Embeddings()
        
        # Test texts
        texts = [
            "Hello, how can I help you today?",
            "I'm having an issue with my order.",
            "Let me check the status of your order."
        ]
        
        print("Generating embeddings for test texts...")
        # Generate embeddings
        embeddings = embeddings_model.generate_embeddings(texts)
        
        # Check if embeddings were generated correctly
        if len(embeddings) != len(texts):
            print(f"Error: Expected {len(texts)} embeddings, but got {len(embeddings)}.")
            return False
        
        # Check if embeddings have the expected shape
        for i, embedding in enumerate(embeddings):
            print(f"Embedding {i+1} shape: {embedding.shape}")
            
        # Test similarity calculation
        print("\nCalculating similarities between embeddings...")
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        print("Similarity matrix:")
        for row in similarities:
            print([f"{x:.4f}" for x in row])
            
        # Test query functionality
        print("\nTesting query functionality...")
        query_text = "order status"
        print(f"Query: '{query_text}'")
        
        # Generate embedding for query
        query_embedding = embeddings_model.generate_embeddings([query_text])[0]
        
        # Calculate similarities with test texts
        query_similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        print("Similarities with test texts:")
        for i, (text, similarity) in enumerate(zip(texts, query_similarities)):
            print(f"{i+1}. '{text}' - Similarity: {similarity:.4f}")
            
        # Find most similar text
        most_similar_idx = np.argmax(query_similarities)
        print(f"\nMost similar text to query: '{texts[most_similar_idx]}'")
        
        return True
        
    except Exception as e:
        print(f"Exception occurred: {e}")
        return False

if __name__ == "__main__":
    print("Testing embeddings module...")
    success = test_embeddings()
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed.")
        sys.exit(1)
