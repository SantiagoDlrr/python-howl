class EmbeddingQueryResult:
    """
    Class to represent the result of an embedding query
    """
    
    def __init__(self, id: str, score: float, metadata: dict = None):
        """
        Initialize the embedding query result
        
        Args:
            id (str): The ID of the document
            score (float): The similarity score
            metadata (dict, optional): Additional metadata. Defaults to None.
        """
        self.id = id
        self.score = score
        self.metadata = metadata or {}