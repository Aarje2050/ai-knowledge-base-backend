import hashlib
import json
import asyncio
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import OrderedDict
from openai import AsyncOpenAI

from app.config import settings
from app.models.schemas import QueryResult

class EnterpriseMemoryCache:
    """
    Enterprise-grade in-memory cache with TTL, LRU eviction, and thread safety.
    Designed for production use without external dependencies.
    """
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0
        }
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = []
        
        for key, item in self._cache.items():
            if item['expires'] < now:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self._stats['evictions'] += 1
    
    def _evict_lru(self):
        """Evict least recently used items when cache is full"""
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest item
            self._stats['evictions'] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            self._cleanup_expired()
            
            if key in self._cache:
                item = self._cache[key]
                if item['expires'] > datetime.now():
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._stats['hits'] += 1
                    return item['value']
                else:
                    del self._cache[key]
                    self._stats['evictions'] += 1
            
            self._stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in cache"""
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            expires = datetime.now() + timedelta(seconds=ttl)
            
            # Remove old entry if exists
            if key in self._cache:
                del self._cache[key]
            
            # Evict if necessary
            self._evict_lru()
            
            # Add new entry
            self._cache[key] = {
                'value': value,
                'expires': expires,
                'created': datetime.now()
            }
            
            self._stats['sets'] += 1
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            self._cleanup_expired()
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': round(hit_rate, 2),
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'sets': self._stats['sets'],
                'memory_usage_mb': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB (rough approximation)"""
        try:
            # Rough estimation: 1KB per cache entry on average
            return len(self._cache) * 1024 / (1024 * 1024)
        except:
            return 0.0

class AdvancedAIService:
    """
    Enterprise-grade AI service with intelligent caching, model routing, and performance optimization.
    Designed for production deployment without external dependencies.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        
        # Model routing configuration
        self.models = {
            "fast": "gpt-3.5-turbo",      # 2-3 seconds, good for simple queries
            "balanced": "gpt-4o-mini",     # 4-6 seconds, good balance
            "accurate": "gpt-4o"           # 8-12 seconds, best quality
        }
        
        # Enterprise memory cache
        self.embedding_cache = EnterpriseMemoryCache(max_size=5000, default_ttl=86400)  # 24 hours
        self.response_cache = EnterpriseMemoryCache(max_size=2000, default_ttl=3600)    # 1 hour
        
        print("✅ Enterprise AI Service initialized with in-memory caching")
    
    async def initialize_cache(self):
        """Initialize cache system (compatibility method)"""
        print("✅ Enterprise memory cache system ready")
        print(f"📊 Embedding cache: {self.embedding_cache.max_size} entries max")
        print(f"📊 Response cache: {self.response_cache.max_size} entries max")
    
    def get_cache_key(self, prefix: str, content: str) -> str:
        """Generate cache key from content"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}:{content_hash}"
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding with intelligent caching"""
        # Check cache first
        cache_key = self.get_cache_key("embedding", text)
        cached_embedding = self.embedding_cache.get(cache_key)
        
        if cached_embedding:
            return cached_embedding
        
        try:
            # Generate new embedding
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text.replace("\n", " ")
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            self.embedding_cache.set(cache_key, embedding, 86400)  # 24 hours
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            raise
    
    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings in batches with intelligent caching"""
        all_embeddings = []
        
        # Process in batches to avoid rate limits and memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache for each text in batch
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                cache_key = self.get_cache_key("embedding", text)
                cached = self.embedding_cache.get(cache_key)
                
                if cached:
                    batch_embeddings.append(cached)
                else:
                    batch_embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    cleaned_texts = [text.replace("\n", " ") for text in uncached_texts]
                    response = await self.client.embeddings.create(
                        model=self.embedding_model,
                        input=cleaned_texts
                    )
                    
                    new_embeddings = [data.embedding for data in response.data]
                    
                    # Cache new embeddings and fill placeholders
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        batch_embeddings[idx] = embedding
                        original_text = uncached_texts[uncached_indices.index(idx)]
                        cache_key = self.get_cache_key("embedding", original_text)
                        self.embedding_cache.set(cache_key, embedding, 86400)  # 24 hours
                        
                except Exception as e:
                    print(f"❌ Error getting batch embeddings: {e}")
                    # Fill remaining placeholders with dummy embeddings
                    dummy_embedding = [0.0] * 1536
                    for idx in uncached_indices:
                        if batch_embeddings[idx] is None:
                            batch_embeddings[idx] = dummy_embedding
            
            all_embeddings.extend(batch_embeddings)
            
            # Add small delay between batches to respect rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return all_embeddings
    
    def classify_query_complexity(self, query: str, context: str) -> str:
        """Classify query complexity for intelligent model routing"""
        
        # Simple queries - use fast model
        simple_indicators = [
            len(query.split()) <= 5,
            query.lower().startswith(('what is', 'what are', 'who is', 'when is')),
            'definition' in query.lower(),
            len(context) < 500,
        ]
        
        # Complex queries - use accurate model
        complex_indicators = [
            'compare' in query.lower(),
            'analyze' in query.lower(),
            'summarize' in query.lower(),
            'explain' in query.lower() and len(query.split()) > 8,
            len(context) > 2000,
            'why' in query.lower() and len(query.split()) > 6,
        ]
        
        # Exact data queries - use balanced model
        exact_indicators = [
            any(term in query.lower() for term in ['cin', 'phone', 'number', 'email', 'code']),
            'what is the' in query.lower() and any(term in query.lower() for term in ['number', 'code', 'id']),
        ]
        
        simple_score = sum(simple_indicators)
        complex_score = sum(complex_indicators)
        exact_score = sum(exact_indicators)
        
        if exact_score > 0:
            return "balanced"  # Good balance for exact data
        elif complex_score >= 2:
            return "accurate"  # Use best model for complex queries
        elif simple_score >= 2:
            return "fast"     # Use fast model for simple queries
        else:
            return "balanced" # Default to balanced
    
    async def generate_response(self, 
                              query: str, 
                              context_results: List[QueryResult],
                              conversation_history: List[Dict] = None,
                              force_model: Optional[str] = None) -> str:
        """Generate response with intelligent model routing and caching"""
        
        # Prepare context
        context_parts = []
        for i, result in enumerate(context_results):
            source_info = ""
            if "filename" in result.metadata:
                source_info = f"Source: {result.metadata['filename']}"
                if "page" in result.metadata:
                    source_info += f" (Page {result.metadata['page']})"
                    
                # Add search type info
                search_type = result.metadata.get('search_type', 'vector')
                if search_type == 'exact_pattern':
                    source_info += f" [Exact Match]"
                elif search_type == 'structured_data':
                    source_info += f" [Structured Data]"
            
            context_parts.append(f"[Context {i+1}]\n{source_info}\nContent: {result.content}\n")
        
        context = "\n".join(context_parts)
        
        # Check response cache
        cache_context = f"{query}|{context[:500]}"  # Truncate context for cache key
        cache_key = self.get_cache_key("response", cache_context)
        cached_response = self.response_cache.get(cache_key)
        
        if cached_response:
            print("🚀 Cache hit - returning cached response")
            return cached_response
        
        # Determine optimal model
        if force_model:
            model_type = force_model
        else:
            model_type = self.classify_query_complexity(query, context)
        
        model_name = self.models[model_type]
        print(f"🤖 Using {model_type} model ({model_name}) for query")
        
        # Build system message with enhanced instructions
        system_message = f"""You are an AI assistant helping users find information from their knowledge base.

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Be specific and accurate
3. Include source references in your response
4. If the context doesn't contain enough information, say so clearly
5. For exact data (numbers, codes, IDs), quote them precisely
6. For policy questions, provide complete relevant information
7. If multiple sources provide conflicting information, mention this

RESPONSE GUIDELINES:
- Keep responses concise but complete
- Use bullet points for lists when appropriate
- Include relevant context from source documents
- Always cite your sources with document names
- If asked for specific data (like CIN numbers), include the exact value

CONTEXT:
{context}
"""

        # Build messages
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history if provided
        if conversation_history:
            # Only include last 4 exchanges to stay within context limits
            recent_history = conversation_history[-8:]  # 4 exchanges = 8 messages
            messages.extend(recent_history)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        try:
            # Generate response with model-specific parameters
            model_params = self.get_model_parameters(model_type)
            
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **model_params
            )
            
            response_text = response.choices[0].message.content
            
            # Cache the response
            self.response_cache.set(cache_key, response_text, 3600)  # 1 hour
            
            return response_text
            
        except Exception as e:
            print(f"❌ Error generating response with {model_name}: {e}")
            
            # Fallback to simpler model if primary fails
            if model_type != "fast":
                print("🔄 Falling back to fast model")
                return await self.generate_response(
                    query, context_results, conversation_history, force_model="fast"
                )
            else:
                return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def get_model_parameters(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific parameters for optimization"""
        if model_type == "fast":
            return {
                "max_tokens": 800,
                "temperature": 0.1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0
            }
        elif model_type == "balanced":
            return {
                "max_tokens": 1000,
                "temperature": 0.1,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1
            }
        else:  # accurate
            return {
                "max_tokens": 1500,
                "temperature": 0.05,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.1
            }
    
    async def classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Enhanced query intent classification"""
        query_lower = query.lower()
        
        # Foundation knowledge indicators
        foundation_keywords = [
            "industry standard", "best practice", "typical", "standard", "common",
            "regulation", "compliance", "legal requirement", "policy template",
            "hr policy", "employee handbook", "vacation policy", "remote work",
            "gdpr", "data protection", "safety", "harassment", "diversity",
            "what should", "how should", "standard procedure"
        ]
        
        # Company-specific indicators
        company_keywords = [
            "our", "we", "company", "internal", "specific", "custom",
            "uploaded", "document", "file", "my company", "this organization"
        ]
        
        # Exact data indicators
        exact_data_keywords = [
            "cin", "phone", "email", "pan", "gst", "ifsc", "account", "registration",
            "number", "code", "id", "contact", "address", "what is the"
        ]
        
        # Comparison indicators
        comparison_keywords = [
            "compare", "vs", "versus", "difference", "benchmark", "contrast",
            "how does our", "how do we compare", "what's the difference"
        ]
        
        # Score each category
        foundation_score = sum(1 for keyword in foundation_keywords if keyword in query_lower)
        company_score = sum(1 for keyword in company_keywords if keyword in query_lower)
        exact_data_score = sum(1 for keyword in exact_data_keywords if keyword in query_lower)
        comparison_score = sum(1 for keyword in comparison_keywords if keyword in query_lower)
        
        return {
            "search_foundation": foundation_score > 0 or (foundation_score >= company_score and comparison_score == 0),
            "search_company": company_score > 0 or (foundation_score == 0 and exact_data_score == 0),
            "needs_exact_search": exact_data_score > 0,
            "needs_comparison": comparison_score > 0,
            "foundation_priority": foundation_score > company_score,
            "query_type": self.determine_primary_intent(foundation_score, company_score, exact_data_score, comparison_score),
            "confidence": max(foundation_score, company_score, exact_data_score, comparison_score) / max(len(query.split()), 1)
        }
    
    def determine_primary_intent(self, foundation_score: int, company_score: int, 
                               exact_data_score: int, comparison_score: int) -> str:
        """Determine the primary intent of the query"""
        scores = {
            "comparison": comparison_score,
            "exact_data": exact_data_score,
            "company_specific": company_score,
            "foundation": foundation_score
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "general"
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics for monitoring"""
        embedding_stats = self.embedding_cache.get_stats()
        response_stats = self.response_cache.get_stats()
        
        return {
            "cache_type": "enterprise_memory",
            "embedding_cache": embedding_stats,
            "response_cache": response_stats,
            "total_memory_mb": embedding_stats['memory_usage_mb'] + response_stats['memory_usage_mb'],
            "overall_hit_rate": round(
                (embedding_stats['hits'] + response_stats['hits']) / 
                max((embedding_stats['hits'] + embedding_stats['misses'] + 
                     response_stats['hits'] + response_stats['misses']), 1) * 100, 2
            )
        }
    
    def clear_cache(self):
        """Clear all caches (useful for testing or memory management)"""
        self.embedding_cache.clear()
        self.response_cache.clear()
        print("🧹 All caches cleared")