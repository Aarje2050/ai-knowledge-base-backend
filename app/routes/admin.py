"""
Admin Routes for Document Management
Add this to your FastAPI app to manage documents via API endpoints.

Save this as: app/routes/admin.py
"""

from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings

router = APIRouter(prefix="/admin", tags=["admin"])

# Initialize Qdrant client
if hasattr(settings, 'qdrant_url') and settings.qdrant_url:
    qdrant_client = QdrantClient(url=settings.qdrant_url)
else:
    qdrant_client = QdrantClient(host="localhost", port=6333)

collection_name = getattr(settings, 'qdrant_collection_name', 'documents')

@router.get("/status")
async def get_qdrant_status():
    """
    Get Qdrant connection status and collection information
    """
    try:
        collections = qdrant_client.get_collections()
        
        # Check if our collection exists
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        result = {
            "qdrant_accessible": True,
            "total_collections": len(collections.collections),
            "target_collection": collection_name,
            "collection_exists": collection_exists
        }
        
        if collection_exists:
            collection_info = qdrant_client.get_collection(collection_name)
            result.update({
                "points_count": collection_info.points_count,
                "vector_dimension": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "status": collection_info.status.name
            })
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to Qdrant: {str(e)}"
        )

@router.get("/documents")
async def list_documents(
    limit: int = Query(100, ge=1, le=1000, description="Number of documents to fetch"),
    company_id: Optional[str] = Query(None, description="Filter by company ID"),
    filename: Optional[str] = Query(None, description="Filter by filename")
):
    """
    List all documents with optional filtering
    """
    try:
        # Build filter if needed
        scroll_filter = None
        if company_id or filename:
            conditions = []
            
            if company_id:
                conditions.append(
                    models.FieldCondition(
                        key="company_id",
                        match=models.MatchValue(value=company_id)
                    )
                )
            
            if filename:
                conditions.append(
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchText(text=filename)
                    )
                )
            
            scroll_filter = models.Filter(must=conditions)
        
        # Fetch documents
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=limit,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False
        )
        
        # Format response
        documents = []
        for point in points:
            payload = point.payload or {}
            documents.append({
                "id": str(point.id),
                "filename": payload.get("filename"),
                "company_id": payload.get("company_id"),
                "page": payload.get("page"),
                "chunk_index": payload.get("chunk_index"),
                "created_at": payload.get("created_at"),
                "content_preview": (payload.get("content", "")[:200] + "...") if payload.get("content") else None,
                "metadata": payload
            })
        
        return {
            "documents": documents,
            "count": len(documents),
            "has_more": next_page_offset is not None,
            "filters_applied": {
                "company_id": company_id,
                "filename": filename
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch documents: {str(e)}"
        )

@router.get("/documents/search")
async def search_documents(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, ge=1, le=200, description="Number of results")
):
    """
    Search documents by content or metadata
    """
    try:
        # Search by content or filename
        search_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="filename",
                    match=models.MatchText(text=query)
                ),
                models.FieldCondition(
                    key="content",
                    match=models.MatchText(text=query)
                )
            ]
        )
        
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        results = []
        for point in points:
            payload = point.payload or {}
            results.append({
                "id": str(point.id),
                "filename": payload.get("filename"),
                "company_id": payload.get("company_id"),
                "page": payload.get("page"),
                "chunk_index": payload.get("chunk_index"),
                "content_preview": (payload.get("content", "")[:300] + "...") if payload.get("content") else None,
                "relevance_score": "text_match"  # Could be enhanced with semantic search
            })
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str = Path(..., description="Document ID to delete")
):
    """
    Delete a specific document by ID
    """
    try:
        # Verify document exists first
        try:
            point = qdrant_client.retrieve(
                collection_name=collection_name,
                ids=[document_id],
                with_payload=True
            )
            if not point:
                raise HTTPException(status_code=404, detail="Document not found")
        except Exception:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete the document
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=[document_id])
        )
        
        return {
            "message": f"Document {document_id} deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.delete("/documents/by-file/{filename}")
async def delete_file(
    filename: str = Path(..., description="Filename to delete (all chunks)")
):
    """
    Delete all chunks of a specific file
    """
    try:
        # Create filter for the filename
        delete_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="filename",
                    match=models.MatchValue(value=filename)
                )
            ]
        )
        
        # Count documents before deletion
        points_before, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=delete_filter,
            limit=1,
            with_payload=False,
            with_vectors=False
        )
        
        if not points_before:
            raise HTTPException(status_code=404, detail=f"No documents found for file: {filename}")
        
        # Delete all chunks
        result = qdrant_client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=delete_filter)
        )
        
        return {
            "message": f"All chunks of file '{filename}' deleted successfully",
            "filename": filename,
            "operation_id": result.operation_id if hasattr(result, 'operation_id') else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )

@router.delete("/documents/by-company/{company_id}")
async def delete_company_documents(
    company_id: str = Path(..., description="Company ID to delete all documents for")
):
    """
    Delete all documents for a specific company
    """
    try:
        # Create filter for the company
        delete_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="company_id",
                    match=models.MatchValue(value=company_id)
                )
            ]
        )
        
        # Count documents before deletion
        points_before, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=delete_filter,
            limit=1,
            with_payload=False,
            with_vectors=False
        )
        
        if not points_before:
            raise HTTPException(status_code=404, detail=f"No documents found for company: {company_id}")
        
        # Delete all company documents
        result = qdrant_client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=delete_filter)
        )
        
        return {
            "message": f"All documents for company '{company_id}' deleted successfully",
            "company_id": company_id,
            "operation_id": result.operation_id if hasattr(result, 'operation_id') else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete company documents: {str(e)}"
        )

@router.delete("/collection")
async def delete_entire_collection():
    """
    ⚠️ DANGER: Delete the entire collection (all documents for all companies)
    Use with extreme caution!
    """
    try:
        # Get current collection info
        collection_info = qdrant_client.get_collection(collection_name)
        points_count = collection_info.points_count
        
        # Delete the entire collection
        qdrant_client.delete_collection(collection_name)
        
        return {
            "message": f"Collection '{collection_name}' deleted successfully",
            "collection_name": collection_name,
            "documents_deleted": points_count,
            "warning": "All documents for all companies have been permanently deleted"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection: {str(e)}"
        )

@router.get("/statistics")
async def get_statistics():
    """
    Get detailed statistics about documents and collections
    """
    try:
        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        
        # Get sample of documents for analysis
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,  # Sample size for analysis
            with_payload=True,
            with_vectors=False
        )
        
        # Analyze the sample
        companies = {}
        files = {}
        upload_dates = []
        total_content_length = 0
        
        for point in points:
            payload = point.payload or {}
            
            # Count by company
            company_id = payload.get('company_id', 'Unknown')
            companies[company_id] = companies.get(company_id, 0) + 1
            
            # Count by file
            filename = payload.get('filename', 'Unknown')
            files[filename] = files.get(filename, 0) + 1
            
            # Collect upload dates
            upload_date = payload.get('created_at')
            if upload_date:
                upload_dates.append(upload_date)
            
            # Content length analysis
            content = payload.get('content', '')
            total_content_length += len(content)
        
        # Calculate statistics
        avg_content_length = total_content_length / len(points) if points else 0
        
        return {
            "collection": {
                "name": collection_name,
                "total_points": collection_info.points_count,
                "vector_dimension": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "status": collection_info.status.name
            },
            "sample_analysis": {
                "sample_size": len(points),
                "companies": {
                    "count": len(companies),
                    "distribution": dict(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10])
                },
                "files": {
                    "count": len(files),
                    "top_files": dict(sorted(files.items(), key=lambda x: x[1], reverse=True)[:10])
                },
                "content": {
                    "avg_chunk_length": round(avg_content_length, 2),
                    "total_content_chars": total_content_length
                },
                "timeline": {
                    "earliest_upload": min(upload_dates) if upload_dates else None,
                    "latest_upload": max(upload_dates) if upload_dates else None
                }
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """
    Simple health check for the admin endpoints
    """
    try:
        # Test Qdrant connection
        qdrant_client.get_collections()
        
        return {
            "status": "healthy",
            "qdrant_connection": "ok",
            "collection_name": collection_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "qdrant_connection": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }