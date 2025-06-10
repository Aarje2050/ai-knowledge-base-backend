from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentUpload(BaseModel):
    filename: str
    content_type: str
    size: int

class DocumentInfo(BaseModel):
    id: str
    filename: str
    content_type: str
    size: int
    processed: bool
    uploaded_at: datetime
    chunk_count: Optional[int] = None

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    session_id: str
    timestamp: datetime

class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class QueryRequest(BaseModel):
    query: str
    company_id: Optional[str] = None
    top_k: int = 5

class QueryResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]