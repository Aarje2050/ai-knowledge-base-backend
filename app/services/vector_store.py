import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models

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
                        "source_type": chunk.get("source_type", "custom")  # foundation or custom
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✅ Added {len(points)} chunks to vector store")
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return False
    
    async def search(self, 
                    query_vector: List[float], 
                    top_k: int = 5,
                    company_id: Optional[str] = None,
                    source_type: Optional[str] = None) -> List[QueryResult]:
        """Search for similar documents"""
        try:
            # Build filter if needed
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
            
            query_filter = None
            if filter_conditions:
                query_filter = Filter(must=filter_conditions)
            
            # Search with lower score threshold for better results
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=0.1,  # Very low threshold to catch more results
                with_payload=True,
                with_vectors=False
            )
            
            # Convert to QueryResult objects
            results = []
            for hit in search_result:
                result = QueryResult(
                    content=hit.payload["content"],
                    score=hit.score,
                    metadata=hit.payload["metadata"]
                )
                results.append(result)
            
            print(f"🔍 Vector search found {len(results)} results")
            if results:
                print(f"🔍 Best score: {results[0].score:.3f}")
                print(f"🔍 Worst score: {results[-1].score:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching vectors: {e}")
            return []
    
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