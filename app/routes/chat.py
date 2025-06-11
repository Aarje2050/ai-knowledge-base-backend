from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import uuid
import time
from datetime import datetime

from app.models.schemas import ChatMessage, ChatResponse
from app.services.ai_service import AdvancedAIService
from app.services.vector_store import AdvancedVectorStore

router = APIRouter()
ai_service = AdvancedAIService()
vector_store = AdvancedVectorStore()

# Simple in-memory conversation storage for Phase 1
conversations: Dict[str, List[Dict]] = {}

@router.on_event("startup")
async def startup_chat_service():
    """Initialize advanced services on startup"""
    await ai_service.initialize_cache()
    print("✅ Advanced chat services initialized")

@router.post("/query", response_model=ChatResponse)
async def advanced_chat_query(message: ChatMessage):
    """Advanced chat query with hybrid search and intelligent routing"""
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = message.session_id or str(uuid.uuid4())
        
        # Get conversation history
        conversation_history = conversations.get(session_id, [])
        
        print(f"🔍 Processing advanced query: {message.message}")
        
        # Analyze query intent
        query_intent = await ai_service.classify_query_intent(message.message)
        print(f"🧠 Query analysis: {query_intent['query_type']} (confidence: {query_intent['confidence']:.2f})")
        
        # Generate embedding for the query
        embedding_start = time.time()
        query_embedding = await ai_service.get_embedding(message.message)
        embedding_time = time.time() - embedding_start
        print(f"⚡ Generated embedding in {embedding_time:.2f}s")
        
        # Determine search strategy based on intent
        search_filters = {"company_id": "default"}  # Phase 1: single tenant
        
        # Advanced hybrid search
        search_start = time.time()
        
        if query_intent["needs_comparison"]:
            # Handle comparison queries
            search_results = await handle_comparison_query(
                message.message, query_embedding, query_intent
            )
        elif query_intent["search_foundation"] and query_intent["search_company"]:
            # Search both foundation and company knowledge
            search_results = await search_hybrid_knowledge(
                message.message, query_embedding, query_intent
            )
        elif query_intent["search_foundation"]:
            # Search only foundation knowledge
            search_filters["source_type"] = "foundation"
            search_results = await vector_store.hybrid_search(
                message.message, query_embedding, top_k=5, filters=search_filters
            )
        else:
            # Search company documents with hybrid approach
            search_results = await vector_store.hybrid_search(
                message.message, query_embedding, top_k=5, filters=search_filters
            )
        
        search_time = time.time() - search_start
        print(f"🔍 Hybrid search completed in {search_time:.2f}s, found {len(search_results)} results")
        
        # Log search results for debugging
        for i, result in enumerate(search_results[:3]):
            search_type = result.metadata.get('search_type', 'unknown')
            print(f"  Result {i+1}: {search_type} search, score {result.score:.3f}")
        
        # Generate response
        if not search_results:
            response_text = self.generate_no_results_response(query_intent)
            sources = []
        else:
            response_start = time.time()
            
            # Use intelligent model routing
            response_text = await ai_service.generate_response(
                query=message.message,
                context_results=search_results,
                conversation_history=conversation_history
            )
            
            response_time = time.time() - response_start
            print(f"🤖 Generated response in {response_time:.2f}s")
            
            # Format sources with enhanced metadata
            sources = format_enhanced_sources(search_results)
        
        # Update conversation history
        conversations[session_id] = conversation_history + [
            {"role": "user", "content": message.message},
            {"role": "assistant", "content": response_text}
        ]
        
        # Keep only last 10 messages per session
        if len(conversations[session_id]) > 10:
            conversations[session_id] = conversations[session_id][-10:]
        
        total_time = time.time() - start_time
        print(f"✅ Total query processing time: {total_time:.2f}s")
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            session_id=session_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        print(f"❌ Error processing advanced query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

async def handle_comparison_query(query: str, query_embedding: List[float], 
                                query_intent: Dict) -> List:
    """Handle document comparison queries"""
    
    # Search both foundation and company knowledge
    foundation_results = await vector_store.hybrid_search(
        query, query_embedding, top_k=3, 
        filters={"source_type": "foundation"}
    )
    
    company_results = await vector_store.hybrid_search(
        query, query_embedding, top_k=3,
        filters={"company_id": "default", "source_type": "custom"}
    )
    
    # Mark results with their source type for comparison
    for result in foundation_results:
        result.metadata["comparison_source"] = "industry_standard"
    
    for result in company_results:
        result.metadata["comparison_source"] = "company_specific"
    
    # Combine and return results
    all_results = foundation_results + company_results
    return sorted(all_results, key=lambda x: x.score, reverse=True)[:5]

async def search_hybrid_knowledge(query: str, query_embedding: List[float], 
                                query_intent: Dict) -> List:
    """Search both foundation and company knowledge intelligently"""
    
    # Determine the split based on query intent
    if query_intent["foundation_priority"]:
        foundation_count = 3
        company_count = 2
    else:
        foundation_count = 2
        company_count = 3
    
    # Search foundation knowledge
    foundation_results = await vector_store.hybrid_search(
        query, query_embedding, top_k=foundation_count,
        filters={"source_type": "foundation"}
    )
    
    # Search company knowledge
    company_results = await vector_store.hybrid_search(
        query, query_embedding, top_k=company_count,
        filters={"company_id": "default", "source_type": "custom"}
    )
    
    # Combine and rank results
    all_results = foundation_results + company_results
    return sorted(all_results, key=lambda x: x.score, reverse=True)

def generate_no_results_response(query_intent: Dict) -> str:
    """Generate contextual no-results response based on query intent"""
    
    if query_intent["needs_exact_search"]:
        return "I couldn't find the specific data you're looking for (like CIN numbers, phone numbers, etc.) in the knowledge base. Please make sure the document containing this information has been uploaded and processed."
    
    elif query_intent["search_foundation"]:
        return "I couldn't find relevant information about industry standards or best practices for your question. The foundation knowledge base might not cover this specific topic yet."
    
    elif query_intent["needs_comparison"]:
        return "I couldn't find enough information to make a comparison. Please ensure both company documents and relevant industry standards are available in the knowledge base."
    
    else:
        return "I couldn't find relevant information in the knowledge base to answer your question. Please make sure related documents have been uploaded and processed."

def format_enhanced_sources(search_results: List) -> List[Dict]:
    """Format sources with enhanced metadata and search type information"""
    sources = []
    
    for i, result in enumerate(search_results):
        # Get search type and add appropriate indicators
        search_type = result.metadata.get('search_type', 'vector')
        source_type = result.metadata.get('comparison_source', 
                     'foundation' if result.metadata.get('source_type') == 'foundation' else 'company')
        
        # Create enhanced source info
        source = {
            "id": i + 1,
            "content": result.content[:300] + "..." if len(result.content) > 300 else result.content,
            "score": round(result.score, 3),
            "metadata": result.metadata,
            "search_type": search_type,
            "source_type": source_type
        }
        
        # Add special indicators for different search types
        if search_type == "exact_pattern":
            source["indicator"] = "🎯 Exact Match"
        elif search_type == "structured_data":
            source["indicator"] = "📊 Structured Data"
        elif source_type == "foundation":
            source["indicator"] = "📚 Industry Standard"
        elif search_type == "keyword":
            source["indicator"] = "🔑 Keyword Match"
        else:
            source["indicator"] = "📄 Document Content"
        
        sources.append(source)
    
    return sources

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

@router.get("/debug/advanced")
async def debug_advanced_features():
    """Debug endpoint for advanced features"""
    try:
        # Test hybrid search capabilities
        test_query = "What is a CIN number?"
        query_embedding = await ai_service.get_embedding(test_query)
        
        # Test different search methods
        vector_results = await vector_store.vector_search(query_embedding, top_k=3)
        hybrid_results = await vector_store.hybrid_search(test_query, query_embedding, top_k=3)
        
        # Get cache statistics
        cache_stats = await ai_service.get_cache_stats()
        
        return {
            "status": "success",
            "advanced_features": {
                "hybrid_search": "enabled",
                "semantic_chunking": "enabled",
                "model_routing": "enabled",
                "caching": cache_stats["cache_type"]
            },
            "test_results": {
                "vector_search_results": len(vector_results),
                "hybrid_search_results": len(hybrid_results),
                "hybrid_improvement": len(hybrid_results) - len(vector_results)
            },
            "cache_stats": cache_stats,
            "search_capabilities": [
                "Vector similarity search",
                "Keyword/TF-IDF search", 
                "Exact pattern matching",
                "Structured data search",
                "Multi-source ranking"
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Advanced features debug failed"
        }

@router.get("/debug/performance")
async def debug_performance():
    """Performance debugging endpoint"""
    try:
        start_time = time.time()
        
        # Test embedding performance
        embedding_start = time.time()
        test_embedding = await ai_service.get_embedding("test query for performance")
        embedding_time = time.time() - embedding_start
        
        # Test search performance
        search_start = time.time()
        search_results = await vector_store.hybrid_search(
            "test query", test_embedding, top_k=5
        )
        search_time = time.time() - search_start
        
        total_time = time.time() - start_time
        
        return {
            "status": "success",
            "performance_metrics": {
                "embedding_generation": f"{embedding_time:.3f}s",
                "hybrid_search": f"{search_time:.3f}s", 
                "total_processing": f"{total_time:.3f}s",
                "results_found": len(search_results)
            },
            "optimizations_active": [
                "Response caching",
                "Embedding caching", 
                "Model routing",
                "Batch processing",
                "Hybrid search"
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/test")
async def test_advanced_search(query: str = "What is a CIN number?"):
    """Advanced test endpoint with detailed search analysis"""
    try:
        start_time = time.time()
        
        # Analyze query
        query_intent = await ai_service.classify_query_intent(query)
        
        # Generate embedding
        query_embedding = await ai_service.get_embedding(query)
        
        # Test different search strategies
        vector_results = await vector_store.vector_search(query_embedding, top_k=3)
        hybrid_results = await vector_store.hybrid_search(query, query_embedding, top_k=3)
        
        processing_time = time.time() - start_time
        
        return {
            "query": query,
            "processing_time": f"{processing_time:.3f}s",
            "query_analysis": query_intent,
            "search_comparison": {
                "vector_only": {
                    "results_count": len(vector_results),
                    "best_score": vector_results[0].score if vector_results else 0,
                    "search_types": ["vector"]
                },
                "hybrid_search": {
                    "results_count": len(hybrid_results),
                    "best_score": hybrid_results[0].score if hybrid_results else 0,
                    "search_types": list(set([r.metadata.get('search_type', 'vector') for r in hybrid_results]))
                }
            },
            "results": [
                {
                    "content": r.content[:100] + "...",
                    "score": round(r.score, 3),
                    "search_type": r.metadata.get('search_type', 'vector'),
                    "filename": r.metadata.get('filename', 'unknown')
                } for r in hybrid_results
            ]
        }
        
    except Exception as e:
        return {"error": str(e), "query": query}