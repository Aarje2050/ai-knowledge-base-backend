from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from typing import List, Optional
import os
import time

from app.services.document_processor import AdvancedDocumentProcessor
from app.config import settings

router = APIRouter()
doc_processor = AdvancedDocumentProcessor()

@router.post("/upload")
async def upload_document_advanced(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company_id: str = "default",
    source_type: str = "custom"
):
    """Advanced document upload with enhanced processing"""
    
    start_time = time.time()
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Enhanced file validation
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed: {', '.join(settings.allowed_file_types)}"
        )
    
    # Read file content
    file_content = await file.read()
    
    # Validate file size
    if len(file_content) > settings.max_file_size:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Max size: {settings.max_file_size / 1024 / 1024:.1f}MB"
        )
    
    # Basic file type validation (without python-magic)
    if len(file_content) < 10:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupted")
    
    try:
        print(f"📄 Starting advanced processing for: {file.filename}")
        
        # Save file
        file_path = await doc_processor.save_uploaded_file(file_content, file.filename)
        
        # Process document with advanced methods
        result = await doc_processor.process_document(file_path, file.filename, company_id)
        
        processing_time = time.time() - start_time
        
        if result["success"]:
            return {
                "message": "Document uploaded and processed successfully with advanced methods",
                "document_id": result["document_id"],
                "filename": result["filename"],
                "chunks_processed": result["chunks_processed"],
                "structured_data_found": result.get("structured_data_found", {}),
                "processing_method": result.get("processing_method", "advanced"),
                "processing_time": f"{processing_time:.2f}s",
                "features_used": [
                    "intelligent_chunking",
                    "structured_data_extraction", 
                    "advanced_text_extraction",
                    "hybrid_search_indexing"
                ]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/")
async def list_documents_advanced(
    company_id: str = "default",
    source_type: Optional[str] = None,
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """List all uploaded documents with advanced filtering"""
    try:
        # Get documents from vector store
        points_response = doc_processor.vector_store.client.scroll(
            collection_name=doc_processor.vector_store.collection_name,
            limit=1000,  # Get more than needed for filtering
            with_payload=True,
            with_vectors=False
        )
        
        points = points_response[0] if points_response else []
        
        # Group by document and apply filters
        documents = {}
        for point in points:
            metadata = point.payload.get("metadata", {})
            doc_company_id = metadata.get("company_id", "unknown")
            doc_source_type = point.payload.get("source_type", "unknown")
            
            # Apply filters
            if company_id != "all" and doc_company_id != company_id:
                continue
            if source_type and doc_source_type != source_type:
                continue
            
            doc_id = point.payload.get("document_id")
            filename = metadata.get("filename", "Unknown")
            
            if doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id,
                    "filename": filename,
                    "company_id": doc_company_id,
                    "source_type": doc_source_type,
                    "chunks": 0,
                    "structured_data": {},
                    "processing_method": metadata.get("chunk_method", "unknown"),
                    "upload_date": metadata.get("upload_date", "unknown")
                }
            
            documents[doc_id]["chunks"] += 1
            
            # Collect structured data info
            if point.payload.get("chunk_type") == "structured_data":
                data_type = metadata.get("data_type", "unknown")
                if data_type not in documents[doc_id]["structured_data"]:
                    documents[doc_id]["structured_data"][data_type] = 0
                documents[doc_id]["structured_data"][data_type] += 1
        
        # Convert to list and apply pagination
        documents_list = list(documents.values())
        documents_list.sort(key=lambda x: x["filename"])
        
        # Apply pagination
        total_count = len(documents_list)
        paginated_docs = documents_list[offset:offset + limit]
        
        return {
            "documents": paginated_docs,
            "pagination": {
                "total": total_count,
                "offset": offset,
                "limit": limit,
                "has_next": offset + limit < total_count
            },
            "summary": {
                "total_documents": total_count,
                "source_types": list(set(doc["source_type"] for doc in documents_list)),
                "companies": list(set(doc["company_id"] for doc in documents_list))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@router.get("/{document_id}")
async def get_document_details(document_id: str):
    """Get detailed information about a specific document"""
    try:
        # Get all chunks for this document
        points_response = doc_processor.vector_store.client.scroll(
            collection_name=doc_processor.vector_store.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        points = points_response[0] if points_response else []
        
        # Filter chunks for this document
        document_chunks = [
            point for point in points 
            if point.payload.get("document_id") == document_id
        ]
        
        if not document_chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Analyze document structure
        first_chunk = document_chunks[0]
        metadata = first_chunk.payload.get("metadata", {})
        
        # Count different chunk types
        chunk_types = {}
        structured_data = {}
        
        for chunk in document_chunks:
            chunk_type = chunk.payload.get("chunk_type", "text")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # Collect structured data
            if chunk_type == "structured_data":
                data_type = chunk.payload.get("metadata", {}).get("data_type", "unknown")
                exact_value = chunk.payload.get("metadata", {}).get("exact_value", "")
                
                if data_type not in structured_data:
                    structured_data[data_type] = []
                structured_data[data_type].append(exact_value)
        
        return {
            "document_id": document_id,
            "filename": metadata.get("filename", "Unknown"),
            "company_id": metadata.get("company_id", "unknown"),
            "source_type": first_chunk.payload.get("source_type", "unknown"),
            "total_chunks": len(document_chunks),
            "chunk_analysis": {
                "types": chunk_types,
                "processing_method": metadata.get("chunk_method", "unknown"),
                "avg_chunk_size": sum(len(chunk.payload.get("content", "")) for chunk in document_chunks) // len(document_chunks)
            },
            "structured_data": structured_data,
            "metadata": {
                "source": metadata.get("source", "unknown"),
                "processing_date": metadata.get("processing_date", "unknown"),
                "total_chunks": metadata.get("total_chunks", len(document_chunks))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document details: {str(e)}")

@router.delete("/{document_id}")
async def delete_document_advanced(document_id: str):
    """Delete a document and all its chunks"""
    try:
        success = await doc_processor.vector_store.delete_document(document_id)
        
        if success:
            return {
                "message": "Document deleted successfully",
                "document_id": document_id,
                "features_updated": [
                    "vector_index_updated",
                    "search_index_rebuilt"
                ]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.get("/stats/overview")
async def get_document_stats():
    """Get overview statistics of all documents"""
    try:
        # Get all points from vector store
        points_response = doc_processor.vector_store.client.scroll(
            collection_name=doc_processor.vector_store.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        points = points_response[0] if points_response else []
        
        # Analyze statistics
        documents = set()
        companies = set()
        source_types = set()
        chunk_types = {}
        structured_data_types = set()
        
        total_content_length = 0
        
        for point in points:
            metadata = point.payload.get("metadata", {})
            
            # Collect unique values
            documents.add(point.payload.get("document_id"))
            companies.add(metadata.get("company_id", "unknown"))
            source_types.add(point.payload.get("source_type", "unknown"))
            
            # Count chunk types
            chunk_type = point.payload.get("chunk_type", "text")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # Collect structured data types
            if chunk_type == "structured_data":
                data_type = metadata.get("data_type", "unknown")
                structured_data_types.add(data_type)
            
            # Calculate content statistics
            content = point.payload.get("content", "")
            total_content_length += len(content)
        
        avg_chunk_size = total_content_length // max(len(points), 1)
        
        return {
            "overview": {
                "total_documents": len(documents),
                "total_chunks": len(points),
                "total_companies": len(companies),
                "avg_chunk_size": avg_chunk_size,
                "total_content_mb": round(total_content_length / 1024 / 1024, 2)
            },
            "breakdown": {
                "source_types": list(source_types),
                "chunk_types": chunk_types,
                "structured_data_types": list(structured_data_types),
                "companies": list(companies)
            },
            "performance": {
                "avg_chunks_per_document": round(len(points) / max(len(documents), 1), 1),
                "structured_data_coverage": f"{len(structured_data_types)} types detected"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")