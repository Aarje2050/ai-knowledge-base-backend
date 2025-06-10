from openai import AsyncOpenAI
from typing import List, Dict, Any
from app.config import settings
from app.models.schemas import QueryResult

class AIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            raise
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            # Clean texts
            cleaned_texts = [text.replace("\n", " ") for text in texts]
            
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"❌ Error getting batch embeddings: {e}")
            raise
    
    async def generate_response(self, 
                              query: str, 
                              context_results: List[QueryResult],
                              conversation_history: List[Dict] = None) -> str:
        """Generate response based on query and context"""
        try:
            # Build context from search results
            context_parts = []
            for i, result in enumerate(context_results):
                source_info = ""
                if "filename" in result.metadata:
                    source_info = f"Source: {result.metadata['filename']}"
                    if "page" in result.metadata:
                        source_info += f" (Page {result.metadata['page']})"
                
                context_parts.append(f"[Context {i+1}]\n{source_info}\nContent: {result.content}\n")
            
            context = "\n".join(context_parts)
            
            # Build system message
            system_message = f"""You are an AI assistant helping users find information from their knowledge base. 

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Be specific and accurate
3. Include source references in your response
4. If the context doesn't contain enough information, say so
5. Keep responses concise but complete

CONTEXT:
{context}
"""
            
            # Build messages
            messages = [{"role": "system", "content": system_message}]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-6:])  # Last 6 messages for context
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=1000,
                temperature=0.1,  # Low temperature for accuracy
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent (foundation, custom, comparison)"""
        # Simple rule-based classification for Phase 1
        query_lower = query.lower()
        
        intent = {
            "needs_foundation": False,
            "needs_custom": True,  # Default to custom for Phase 1
            "needs_comparison": False,
            "confidence": 0.8
        }
        
        # Foundation indicators
        foundation_indicators = ["industry standard", "best practice", "regulation", "compliance", "benchmark"]
        if any(indicator in query_lower for indicator in foundation_indicators):
            intent["needs_foundation"] = True
            intent["needs_custom"] = False
        
        # Comparison indicators  
        comparison_indicators = ["compare", "vs", "versus", "difference", "benchmark against"]
        if any(indicator in query_lower for indicator in comparison_indicators):
            intent["needs_comparison"] = True
            intent["needs_foundation"] = True
        
        # Company-specific indicators
        company_indicators = ["our", "we", "company", "internal", "specific"]
        if any(indicator in query_lower for indicator in company_indicators):
            intent["needs_custom"] = True
        
        return intent