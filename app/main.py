from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.config import settings
from app.routes import documents, chat
from app.services.vector_store import VectorStore
from app.services.ai_service import AIService

# Create upload directory
os.makedirs(settings.upload_dir, exist_ok=True)

app = FastAPI(title="AI Knowledge Platform", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
vector_store = VectorStore()
ai_service = AIService()

# Include routes
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

# Mount uploads directory
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await vector_store.initialize()
    print("✅ Vector store initialized")

@app.get("/")
async def root():
    return {"message": "AI Knowledge Platform API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}