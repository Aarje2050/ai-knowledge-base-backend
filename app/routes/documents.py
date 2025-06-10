from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
import os

from app.services.document_processor import DocumentProcessor
from app.config import settings

router = APIRouter()
doc_processor = DocumentProcessor()

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company_id: str = "default"
):
    """Upload and process a document"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    allowed_extensions = {'.pdf', '.docx', '.txt', '.md', '.doc'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > settings.max_file_size:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Max size: {settings.max_file_size / 1024 / 1024}MB"
        )
    
    try:
        # Save file
        file_path = await doc_processor.save_uploaded_file(file_content, file.filename)
        
        # Process document in background
        result = await doc_processor.process_document(file_path, file.filename, company_id)
        
        if result["success"]:
            return {
                "message": "Document uploaded and processed successfully",
                "document_id": result["document_id"],
                "filename": result["filename"],
                "chunks_processed": result["chunks_processed"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/")
async def list_documents():
    """List all uploaded documents (placeholder for Phase 1)"""
    return {"message": "Document listing not implemented in Phase 1"}

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        success = await doc_processor.vector_store.delete_document(document_id)
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))