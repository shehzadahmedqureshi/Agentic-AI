# Optional: Load environment variables if not using pipenv
# from dotenv import load_dotenv
# load_dotenv()

# === Request Schema Definition ===
from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    model: str
    provider: str
    prompt: str
    messages: List[str]
    search_enabled: bool

# === FastAPI Backend Setup ===
from fastapi import FastAPI
from ai_agent import query_agent

SUPPORTED_MODELS = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"]

app = FastAPI(title="Conversational AI Service")

@app.post("/chat")
def handle_chat(request: ChatRequest):
    """
    Endpoint for interacting with the conversational AI agent.
    Validates model and delegates to the agent logic.
    """
    if request.model not in SUPPORTED_MODELS:
        return {"error": "Selected model is not supported. Please choose a valid model."}

    response = query_agent(
        model_name=request.model,
        messages=request.messages,
        enable_search=request.search_enabled,
        prompt=request.prompt,
        provider=request.provider
    )
    return response

# === Local Development Entrypoint ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
