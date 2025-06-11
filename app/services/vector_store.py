import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import asyncio

from app.config import settings
from app.models.schemas import QueryResult

class AdvancedVectorStore:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = "knowledge_base"
        
        # Search patterns for exact matching
        self.exact_patterns = {
            'cin': r'CIN\s*:?\s*[A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}',
            'phone': r'\+?[\d\s\-\(\)]{10,15}',
            'email': r'[\w\.-]+@[\w\.-]+\.\w+',
            'pan': r'PAN\s*:?\s*[A-Z]{5}\d{4}[A-Z]',
            'gst': r'GST\s*:?\s*\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]',
            'ifsc': r'IFSC\s*:?\s*[A-Z]{4}0[A-Z0-9]{6}',
            'account': r'Account\s+(?:No\.?|Number)\s*:?\s*\d{9,18}',
            'registration': r'Registration\s+(?:No\.?|Number)\s*:?\s*[A-Z0-9\-/]+',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        }
        
    async def initialize(self):
        """Initialize Qdrant collection"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
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
                
            print(f"✅ Advanced vector store initialized")
                
        except Exception as e:
            print(f"❌ Error initializing vector store: {e}")
            raise
    
    async def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add document chunks with enhanced indexing"""
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
                        "source_type": chunk.get("source_type", "custom"),
                        # Add searchable fields for exact matching
                        "content_lower": chunk["content"].lower(),
                        "searchable_terms": self.extract_searchable_terms(chunk["content"]),
                        "chunk_type": chunk["metadata"].get("chunk_type", "text")
                    }
                )
                points.append(point)
            
            # Add points in batches for better performance
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f"✅ Added batch {i//batch_size + 1}: {len(batch)} chunks")
            
            print(f"✅ Successfully added all {len(points)} chunks with enhanced indexing")
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return False
    
    def extract_searchable_terms(self, content: str) -> List[str]:
        """Extract searchable terms for exact matching"""
        terms = []
        content_lower = content.lower()
        
        # Extract exact patterns
        for pattern_name, pattern in self.exact_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                terms.append(match.strip())
                # Add normalized version
                normalized = re.sub(r'[^\w]', '', match.strip().lower())
                if normalized:
                    terms.append(normalized)
        
        # Extract important keywords
        important_keywords = ['policy', 'procedure', 'guideline', 'requirement', 'standard', 'compliance']
        for keyword in important_keywords:
            if keyword in content_lower:
                terms.append(keyword)
        
        return list(set(terms))  # Remove duplicates
    
    async def vector_search(self, query_vector: List[float], top_k: int = 10, 
                           filters: Optional[Dict] = None, score_threshold: float = 0.1) -> List[QueryResult]:
        """Traditional vector similarity search"""
        try:
            # Build filter conditions
            query_filter = None
            if filters:
                filter_conditions = []
                
                for key, value in filters.items():
                    if key == "company_id":
                        filter_conditions.append(
                            FieldCondition(key="metadata.company_id", match=MatchValue(value=value))
                        )
                    elif key == "source_type":
                        filter_conditions.append(
                            FieldCondition(key="source_type", match=MatchValue(value=value))
                        )
                    elif key == "chunk_type":
                        filter_conditions.append(
                            FieldCondition(key="chunk_type", match=MatchValue(value=value))
                        )
                
                if filter_conditions:
                    query_filter = Filter(must=filter_conditions)
            
            # Perform vector search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
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
            
            return results
            
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            return []
    
    async def exact_pattern_search(self, query: str) -> List[QueryResult]:
        """Search for exact patterns like CIN numbers, phone numbers"""
        results = []
        
        try:
            # Check if query contains or asks for specific patterns
            query_lower = query.lower()
            relevant_patterns = []
            
            # Determine which patterns to search for
            for pattern_name, pattern in self.exact_patterns.items():
                if (pattern_name in query_lower or 
                    any(keyword in query_lower for keyword in [
                        'cin', 'phone', 'email', 'pan', 'gst', 'ifsc', 'account', 'registration', 'number'
                    ])):
                    relevant_patterns.append((pattern_name, pattern))
            
            # If no specific patterns identified, search all
            if not relevant_patterns:
                relevant_patterns = list(self.exact_patterns.items())
            
            # Get all documents and search for patterns
            points_response = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            points = points_response[0] if points_response else []
            
            for point in points:
                content = point.payload.get("content", "")
                
                # Check for exact pattern matches
                for pattern_name, pattern in relevant_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # High score for exact matches
                        result = QueryResult(
                            content=content,
                            score=0.95,  # Very high score for exact matches
                            metadata={
                                **point.payload.get("metadata", {}),
                                "match_type": "exact_pattern",
                                "pattern_type": pattern_name,
                                "matched_values": matches
                            }
                        )
                        results.append(result)
                        break  # Avoid duplicate results for same document
            
            return results
            
        except Exception as e:
            print(f"❌ Error in exact pattern search: {e}")
            return []
    
    async def keyword_search(self, query: str, top_k: int = 10) -> List[QueryResult]:
        """Simple keyword search without TF-IDF"""
        results = []
        
        try:
            # Get all documents
            points_response = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            points = points_response[0] if points_response else []
            
            # Simple keyword matching
            query_words = set(query.lower().split())
            
            for point in points:
                content = point.payload.get("content", "").lower()
                content_words = set(content.split())
                
                # Calculate simple overlap score
                overlap = len(query_words.intersection(content_words))
                if overlap > 0:
                    score = overlap / len(query_words)
                    
                    result = QueryResult(
                        content=point.payload.get("content", ""),
                        score=score * 0.8,  # Lower weight than vector search
                        metadata=point.payload.get("metadata", {})
                    )
                    results.append(result)
            
            # Sort by score and return top results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"❌ Error in keyword search: {e}")
            return []
    
    async def hybrid_search(self, query: str, query_vector: List[float], 
                           top_k: int = 10, filters: Optional[Dict] = None) -> List[QueryResult]:
        """Advanced hybrid search combining vector, keyword, and exact pattern matching"""
        
        print(f"🔍 Starting hybrid search for: {query}")
        
        # Determine query type and search strategy
        query_analysis = self.analyze_query(query)
        
        all_results = []
        
        # 1. Vector search (always performed)
        vector_results = await self.vector_search(
            query_vector, 
            top_k=max(5, top_k//2), 
            filters=filters,
            score_threshold=0.05  # Lower threshold for better recall
        )
        
        for result in vector_results:
            result.metadata["search_type"] = "vector"
            all_results.append(result)
        
        print(f"📊 Vector search found {len(vector_results)} results")
        
        # 2. Exact pattern search (for specific data queries)
        if query_analysis["needs_exact_search"]:
            exact_results = await self.exact_pattern_search(query)
            for result in exact_results:
                result.metadata["search_type"] = "exact_pattern"
                all_results.append(result)
            
            print(f"🎯 Exact pattern search found {len(exact_results)} results")
        
        # 3. Keyword search (for better recall)
        if query_analysis["needs_keyword_search"]:
            keyword_results = await self.keyword_search(query, top_k=max(3, top_k//3))
            
            for result in keyword_results:
                result.metadata["search_type"] = "keyword"
                all_results.append(result)
            
            print(f"🔑 Keyword search found {len(keyword_results)} results")
        
        # 4. Structured data prioritization
        structured_data_results = [r for r in all_results 
                                 if r.metadata.get("chunk_type") == "structured_data"]
        
        if structured_data_results and query_analysis["needs_exact_search"]:
            print(f"📋 Found {len(structured_data_results)} structured data results - prioritizing")
            # Boost scores for structured data when appropriate
            for result in structured_data_results:
                result.score = min(0.98, result.score + 0.2)
        
        # 5. Deduplicate and rank results
        final_results = self.deduplicate_and_rank_results(all_results, query_analysis)
        
        print(f"✅ Hybrid search returning {len(final_results)} final results")
        
        return final_results[:top_k]
    
    def analyze_query(self, query: str) -> Dict[str, bool]:
        """Analyze query to determine optimal search strategy"""
        query_lower = query.lower()
        
        # Indicators for exact search
        exact_indicators = [
            'cin', 'phone', 'email', 'pan', 'gst', 'ifsc', 'account', 'registration',
            'number', 'code', 'id', 'contact', 'address'
        ]
        
        # Indicators for keyword search
        keyword_indicators = [
            'policy', 'procedure', 'guideline', 'rule', 'requirement', 'standard',
            'compliance', 'regulation', 'process', 'how to', 'what is', 'definition'
        ]
        
        analysis = {
            "needs_exact_search": any(indicator in query_lower for indicator in exact_indicators),
            "needs_keyword_search": (
                any(indicator in query_lower for indicator in keyword_indicators) or
                len(query.split()) > 3  # Longer queries benefit from keyword search
            ),
            "is_specific_data_query": any(indicator in query_lower for indicator in exact_indicators),
            "is_policy_query": any(indicator in query_lower for indicator in keyword_indicators[:6]),
        }
        
        return analysis
    
    def deduplicate_and_rank_results(self, results: List[QueryResult], 
                                   query_analysis: Dict[str, bool]) -> List[QueryResult]:
        """Deduplicate and intelligently rank combined search results"""
        
        # Group results by content similarity to avoid duplicates
        unique_results = []
        seen_content = set()
        
        # Sort by score first
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        for result in sorted_results:
            # Simple deduplication based on content similarity
            content_key = result.content[:100].lower().strip()
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                
                # Apply search type weights
                if result.metadata.get("search_type") == "exact_pattern":
                    result.score = min(0.99, result.score + 0.1)  # Boost exact matches
                elif result.metadata.get("search_type") == "keyword" and query_analysis["is_policy_query"]:
                    result.score = min(0.95, result.score + 0.05)  # Slight boost for policy queries
                
                unique_results.append(result)
        
        # Final sort by adjusted scores
        return sorted(unique_results, key=lambda x: x.score, reverse=True)
    
    async def search(self, query_vector: List[float], top_k: int = 5,
                    company_id: Optional[str] = None, source_type: Optional[str] = None) -> List[QueryResult]:
        """Backward compatible search method"""
        filters = {}
        if company_id:
            filters["company_id"] = company_id
        if source_type:
            filters["source_type"] = source_type
        
        return await self.vector_search(query_vector, top_k, filters)
    
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