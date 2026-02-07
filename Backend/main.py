from pydantic import BaseModel,Field
from typing import List,Dict,Annotated
from fastapi import FastAPI,HTTPException
from Backend.Agents.gr_agent import refiner
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
class request(BaseModel):
    query:Annotated[str,Field(description="enter query ")]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Development ke liye - production mein specific domain dalo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/refiner")
async def prompt_refiner(body:request):
    try:
        async def generate():
            async for token in refiner(body.query):
                yield token
    
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.get("/")
def health():
    return {"status": "ok"}