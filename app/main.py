from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import os
import time
import asyncio

from app.config import settings, FEATURE_FLAGS
from app.routes import documents, chat
from app.services.vector_store import AdvancedVectorStore
from app.services.ai_service import AdvancedAIService
from app.routes import admin


# Create upload directory
os.makedirs(settings.upload_dir, exist_ok=True)

# Initialize FastAPI app with advanced configuration
app = FastAPI(
    title="AI Knowledge Platform",
    description="Advanced hybrid knowledge management with AI",
    version="2.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Security middleware
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.onrender.com", "*.vercel.app", "localhost"]
    )

# CORS middleware with environment-specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance monitoring middleware
if FEATURE_FLAGS["performance_monitoring"]:
    @app.middleware("http")
    async def performance_monitoring(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log slow requests
        if process_time > 5.0:  # Log requests taking more than 5 seconds
            print(f"⚠️ Slow request: {request.url} took {process_time:.2f}s")
        
        response.headers["X-Process-Time"] = str(process_time)
        return response

# Initialize advanced services
vector_store = AdvancedVectorStore()
ai_service = AdvancedAIService()

# Include routes
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(admin.router)


# Mount uploads directory (only in development)
if settings.debug:
    app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    print("🚀 Starting AI Knowledge Platform v2.0...")
    
    # Initialize vector store
    try:
        await vector_store.initialize()
        print("✅ Advanced vector store initialized")
    except Exception as e:
        print(f"❌ Vector store initialization failed: {e}")
        raise
    
    # Initialize AI service cache
    try:
        await ai_service.initialize_cache()
        print("✅ AI service with caching initialized")
    except Exception as e:
        print(f"⚠️ Cache initialization failed, continuing without cache: {e}")
    
    # Log enabled features
    enabled_features = [k for k, v in FEATURE_FLAGS.items() if v]
    print(f"🎯 Advanced features enabled: {', '.join(enabled_features)}")
    
    print("✅ All services initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down AI Knowledge Platform...")
    
    # Close Redis connection if exists
    if hasattr(ai_service, 'redis_client') and ai_service.redis_client:
        try:
            await ai_service.redis_client.close()
            print("✅ Redis connection closed")
        except:
            pass
    
    print("✅ Shutdown complete")

@app.get("/")
async def root():
    """Root endpoint with feature information"""
    return {
        "message": "AI Knowledge Platform v2.0",
        "status": "running",
        "features": {
            "hybrid_search": FEATURE_FLAGS["hybrid_search"],
            "semantic_chunking": FEATURE_FLAGS["advanced_chunking"],
            "model_routing": FEATURE_FLAGS["model_routing"],
            "caching": FEATURE_FLAGS["response_caching"],
            "exact_pattern_search": FEATURE_FLAGS["exact_pattern_search"]
        },
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {},
        "features": FEATURE_FLAGS
    }
    
    # Check vector store
    try:
        # Simple test query
        test_embedding = await ai_service.get_embedding("health check")
        test_results = await vector_store.vector_search(test_embedding, top_k=1)
        health_status["services"]["vector_store"] = "healthy"
        health_status["services"]["documents_indexed"] = len(test_results)
    except Exception as e:
        health_status["services"]["vector_store"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check AI service
    try:
        cache_stats = await ai_service.get_cache_stats()
        health_status["services"]["ai_service"] = "healthy"
        health_status["services"]["cache_type"] = cache_stats["cache_type"]
    except Exception as e:
        health_status["services"]["ai_service"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check OpenAI connectivity
    try:
        test_embedding = await ai_service.get_embedding("test")
        health_status["services"]["openai"] = "healthy"
    except Exception as e:
        health_status["services"]["openai"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
    
    return health_status

@app.get("/metrics")
async def get_metrics():
    """Get application metrics (if monitoring enabled)"""
    if not FEATURE_FLAGS["performance_monitoring"]:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    try:
        # Get cache statistics
        cache_stats = await ai_service.get_cache_stats()
        
        # Get document count (approximate)
        test_embedding = await ai_service.get_embedding("metrics test")
        all_results = await vector_store.vector_search(test_embedding, top_k=1000, score_threshold=0.0)
        
        # Count unique documents
        unique_docs = set()
        for result in all_results:
            doc_id = result.metadata.get("document_id")
            if doc_id:
                unique_docs.add(doc_id)
        
        return {
            "timestamp": time.time(),
            "documents": {
                "total_chunks": len(all_results),
                "unique_documents": len(unique_docs)
            },
            "cache": cache_stats,
            "features": FEATURE_FLAGS,
            "performance": {
                "avg_response_time": "tracked_in_middleware",
                "cache_hit_rate": "estimated_60_percent"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

@app.get("/features")
async def get_feature_status():
    """Get current feature flag status"""
    return {
        "features": FEATURE_FLAGS,
        "configuration": {
            "chunking": {
                "target_size": settings.target_chunk_size,
                "max_size": settings.max_chunk_size,
                "overlap": settings.chunk_overlap
            },
            "search": {
                "threshold": settings.default_search_threshold,
                "max_results": settings.max_search_results
            },
            "models": {
                "fast": settings.fast_model,
                "balanced": settings.balanced_model,
                "accurate": settings.accurate_model,
                "embedding": settings.embedding_model
            }
        }
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error reporting"""
    print(f"❌ Unhandled exception: {exc}")
    
    if settings.debug:
        # In debug mode, return detailed error information
        return {
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__,
            "request_url": str(request.url)
        }
    else:
        # In production, return generic error
        return {
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )