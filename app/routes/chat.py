from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import uuid
from datetime import datetime

from app.models.schemas import ChatMessage, ChatResponse
from app.services.ai_service import AIService
from app.services.vector_store import VectorStore

router = APIRouter()
ai_service = AIService()
vector_store = VectorStore()

# Simple in-memory conversation storage for Phase 1
conversations: Dict[str, List[Dict]] = {}

@router.post("/query", response_model=ChatResponse)
async def chat_query(message: ChatMessage):
    """Process a chat query"""
    try:
        # Generate session ID if not provided
        session_id = message.session_id or str(uuid.uuid4())
        
        # Get conversation history
        conversation_history = conversations.get(session_id, [])
        
        # Generate embedding for the query
        query_embedding = await ai_service.get_embedding(message.message)
        
        # Search for relevant documents
        search_results = await vector_store.search(
            query_vector=query_embedding,
            top_k=5,
            company_id="default"  # For Phase 1, using default company
        )
        
        if not search_results:
            response_text = "I couldn't find any relevant information in the knowledge base to answer your question. Please make sure documents have been uploaded and processed."
            sources = []
        else:
            # Generate response using AI
            response_text = await ai_service.generate_response(
                query=message.message,
                context_results=search_results,
                conversation_history=conversation_history
            )
            
            # Format sources
            sources = []
            for i, result in enumerate(search_results):
                source = {
                    "id": i + 1,
                    "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "score": round(result.score, 3),
                    "metadata": result.metadata
                }
                sources.append(source)
        
        # Update conversation history
        conversations[session_id] = conversation_history + [
            {"role": "user", "content": message.message},
            {"role": "assistant", "content": response_text}
        ]
        
        # Keep only last 10 messages per session
        if len(conversations[session_id]) > 10:
            conversations[session_id] = conversations[session_id][-10:]
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            session_id=session_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    history = conversations.get(session_id, [])
    return {"session_id": session_id, "history": history}

@router.delete("/sessions/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversations:
        del conversations[session_id]
    return {"message": "Conversation cleared"}

@router.get("/test")
async def test_search(query: str = "test query"):
    """Test endpoint to check if search is working"""
    try:
        # Generate embedding
        query_embedding = await ai_service.get_embedding(query)
        
        # Search
        results = await vector_store.search(query_embedding, top_k=3)
        
        return {
            "query": query,
            "results_found": len(results),
            "results": [
                {
                    "content": r.content[:100] + "...",
                    "score": r.score,
                    "metadata": r.metadata
                } for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))