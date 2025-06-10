import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.config import settings
from app.models.schemas import QueryResult

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = "knowledge_base"
        
    async def initialize(self):
        """Initialize Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created collection: {self.collection_name}")
            else:
                print(f"✅ Collection exists: {self.collection_name}")
                
            # Test the collection by trying to scroll
            test_scroll = self.client.scroll(
                collection_name=self.collection_name,
                limit=1
            )
            print(f"✅ Collection is accessible and working")
                
        except Exception as e:
            print(f"❌ Error initializing vector store: {e}")
            raise
    
    async def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add document chunks to vector store"""
        try:
            points = []
            for chunk in chunks:
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk["embedding"],
                    payload={
                        "document_id": chunk["document_id"],
                        "content": chunk["content"],
                        "metadata": chunk["metadata"],
                        "chunk_index": chunk.get("chunk_index", 0),
                        "source_type": chunk.get("source_type", "custom")
                    }
                )
                points.append(point)
            
            # Add points in batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f"✅ Added batch {i//batch_size + 1}: {len(batch)} chunks")
            
            print(f"✅ Successfully added all {len(points)} chunks to vector store")
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return False
    
    async def search(self, 
                    query_vector: List[float], 
                    top_k: int = 5,
                    company_id: Optional[str] = None,
                    source_type: Optional[str] = None) -> List[QueryResult]:
        """Search for similar documents with robust error handling"""
        try:
            # Build filter conditions
            filter_conditions = []
            
            if company_id:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.company_id",
                        match=MatchValue(value=company_id)
                    )
                )
            
            if source_type:
                filter_conditions.append(
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=source_type)
                    )
                )
            
            # Create filter object if conditions exist
            query_filter = None
            if filter_conditions:
                query_filter = Filter(must=filter_conditions)
            
            # Perform search with robust error handling
            try:
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=0.1,  # Low threshold for better recall
                    with_payload=True,
                    with_vectors=False
                )
            except Exception as search_error:
                print(f"❌ Search with filter failed: {search_error}")
                # Retry without filters
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=0.1,
                    with_payload=True,
                    with_vectors=False
                )
                print(f"✅ Search without filter succeeded")
            
            # Convert to QueryResult objects
            results = []
            for hit in search_result:
                try:
                    # Safely extract content and metadata
                    content = hit.payload.get("content", "No content available")
                    metadata = hit.payload.get("metadata", {})
                    
                    result = QueryResult(
                        content=content,
                        score=hit.score,
                        metadata=metadata
                    )
                    results.append(result)
                except Exception as result_error:
                    print(f"❌ Error processing search result: {result_error}")
                    continue
            
            print(f"🔍 Vector search returned {len(results)} results")
            if results:
                print(f"🔍 Score range: {results[0].score:.3f} to {results[-1].score:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Critical error in vector search: {e}")
            return []
    
    async def count_documents(self) -> int:
        """Count total documents in collection"""
        try:
            # Use scroll to count documents
            points_response = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Get first 1000 to count
                with_payload=False,
                with_vectors=False
            )
            
            points = points_response[0] if points_response else []
            return len(points)
            
        except Exception as e:
            print(f"❌ Error counting documents: {e}")
            return 0
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            print(f"✅ Deleted document: {document_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Qdrant is healthy and accessible"""
        try:
            # Simple health check
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            return {
                "status": "healthy",
                "collections": collection_names,
                "target_collection_exists": self.collection_name in collection_names
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }