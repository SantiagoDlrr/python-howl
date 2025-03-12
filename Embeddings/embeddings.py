""" 
Class to handle embeddings 
"""

from sentence_transformers import SentenceTransformer


class Embeddings():

    def __init__(self, model_name:str = 'all-MiniLM-L12-v2') -> None:
        """ Load the model """
        self.embedding_model = SentenceTransformer(model_name)
        print("Model loaded")

    def generate_embeddings(self, texts):
        return self.embedding_model.encode(texts)
    
if __name__ == "__main__":
    emb = Embeddings()
    print(emb.generate_embeddings(["Hello World", "This is a test"]))