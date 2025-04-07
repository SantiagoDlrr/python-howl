from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from Embeddings.embeddings import Embeddings

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
emb = Embeddings()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def general():
    return {"Message": "Hello World"}

@app.post("/embedding")
def embedding(req: TextRequest):
    text = req.text
    print(f"Received text: {text}")
    embedding = emb.generate_embeddings([text])
    return {"embedding": embedding[0].tolist()}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)