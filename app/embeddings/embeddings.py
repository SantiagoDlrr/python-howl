""" 
Class to handle embeddings, insert and query
"""

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from embedding_query_result import EmbeddingQueryResult

class Embeddings():

    def __init__(self, model_name:str = 'intfloat/e5-large-v2') -> None:
        """ Load the model and initialize Pinecone """
        load_dotenv() 
        pinecone_db = os.getenv("PINECONE_KEY")
        self.embedding_model = SentenceTransformer(model_name)
        self.pc = Pinecone(api_key=pinecone_db)
        self.index = self.pc.Index("howl")
        print("Model loaded")

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """ 
        Generate embeddings for a list of texts 
        Args:
            texts (list): List of texts to generate embeddings for
        Returns:
            list: List of embeddings
        """
        return self.embedding_model.encode(texts)
    
    def save_embedding(self, call_id, id: int, embedding: list[float], text: str) -> None:
        """ 
        Save the embedding to Pinecone 
        Args:
            call_id (str): The ID of the call
            id (int): The ID of the embedding
            embedding (list): The embedding to save
            text (str): The text associated with the embedding
        """
        self.index.upsert([
            (
                f"id-{call_id}-{id}",  # make ID globally unique
                embedding,
                {"text": text, "call_id": str(call_id)}
            )
        ]) 

    def save_embeddings(self, call_id, embeddings: list[list[float]], texts: list[str]) -> None:
        """ 
        Save multiple embeddings to Pinecone 
        Args:
            call_id (str): The ID of the call
            embeddings (list): List of embeddings to save
            texts (list): List of texts associated with the embeddings
        """
        for i, (embedding, text) in enumerate(zip(embeddings, texts)):
            self.save_embedding(
                call_id=call_id,
                id=i,
                embedding=embedding,
                text=text
            )

    def query(self, text, top_k=5):
        """ 
        Query the embedding 
        Args:
            text (str): The text to query
            top_k (int): The number of top results to return
        Returns:
            results: The top_k results from the query
        """
        query_vector = self.embedding_model.encode([text])[0].tolist()  # embed the query
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        json_results = self.parse_results(results.matches)
        return json_results
    
    def parse_results(self, results):
        """ 
        Parse the results from the query 
        Args:
            results: The results from the query
        Returns:
            list: List of parsed results
        """
        parsed_results = []
        for result in results:
            parsed_results.append(
                EmbeddingQueryResult(
                    id=result.id,
                    call_id=result.metadata["call_id"],
                    text=result.metadata["text"],
                    score=result.score
                )
            )

        json_results = [result.model_dump_json() for result in parsed_results]
        return json_results

if __name__ == "__main__":
    """ Example usage """
    emb = Embeddings()
    chunks = [
        "Agent: Hello, how can I help you?",
        "Customer: My database went down after the update.",
        "Agent: I see, let me check our logs...",
    ]

    # Save embeddings example:
    # embeddings = emb.generate_embeddings(chunks)
    # emb.save_embeddings(1, embeddings, chunks)

    query = "Greeting"
    result = emb.query(query, 1)
    print(result)


