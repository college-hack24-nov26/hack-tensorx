from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.rag_chatbot import RAGChatbot
import os
from dotenv import load_dotenv

# Load .env from parent directory if running from backend folder
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    load_dotenv("../.env")

router = APIRouter()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment. Please add it to .env file")

chatbot = RAGChatbot(groq_api_key)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = chatbot.chat(request.message)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/reset")
async def reset_chat():
    try:
        chatbot.reset_memory()
        return {"message": "Chat history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/rebuild")
async def rebuild_vectorstore():
    try:
        import traceback
        chatbot.create_vectorstore()
        return {"message": "Vector database rebuilt successfully"}
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error rebuilding vectorstore: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))
