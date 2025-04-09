from pydantic import BaseModel

class EmbeddingQueryResult(BaseModel):
    id: str
    call_id: str
    text: str
    score: float