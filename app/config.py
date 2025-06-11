import os
from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str
    
    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    
    # Database Configuration
    database_url: str = "postgresql://user:password@localhost:5432/ai_knowledge"
    
    # Redis Configuration (for caching)
    redis_url: Optional[str] = None  # e.g., "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # App Configuration
    secret_key: str = "your-secret-key-here"
    debug: bool = True
    
    # File Upload Configuration
    upload_dir: str = "uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = [".pdf", ".docx", ".doc", ".txt", ".md"]
    
    # Advanced Processing Configuration
    enable_semantic_chunking: bool = True
    enable_hybrid_search: bool = True
    enable_caching: bool = True
    enable_model_routing: bool = True
    
    # Chunking Configuration
    target_chunk_size: int = 800
    max_chunk_size: int = 1200
    chunk_overlap: int = 100
    
    # Search Configuration
    default_search_threshold: float = 0.05
    max_search_results: int = 10
    enable_exact_pattern_search: bool = True
    enable_keyword_search: bool = True
    
    # Model Configuration
    fast_model: str = "gpt-3.5-turbo"
    balanced_model: str = "gpt-4o-mini"
    accurate_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    
    # Cache Configuration
    embedding_cache_ttl: int = 86400  # 24 hours
    response_cache_ttl: int = 3600    # 1 hour
    max_memory_cache_size: int = 1000
    
    # Performance Configuration
    embedding_batch_size: int = 100
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000"]
    
    # Monitoring Configuration
    enable_metrics: bool = True
    enable_query_logging: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Update CORS origins based on environment
if not settings.debug:
    # Production CORS settings
    settings.cors_origins = [
        "https://your-frontend-domain.vercel.app",
        "https://frontend-ai-knowledge-base.vercel.app"
    ]
    
    # Production optimizations
    settings.enable_caching = True
    settings.enable_model_routing = True
    settings.max_concurrent_requests = 50

# Validate critical settings
def validate_settings():
    """Validate critical configuration settings"""
    errors = []
    
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        errors.append("OpenAI API key not configured")
    
    if not settings.qdrant_url:
        errors.append("Qdrant URL not configured")
    
    if settings.max_file_size > 100 * 1024 * 1024:  # 100MB
        errors.append("Max file size too large (>100MB)")
    
    if settings.target_chunk_size > settings.max_chunk_size:
        errors.append("Target chunk size cannot be larger than max chunk size")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

# Validate on import
try:
    validate_settings()
    print("✅ Configuration validated successfully")
except ValueError as e:
    print(f"⚠️ Configuration warning: {e}")

# Feature flags for gradual rollout
FEATURE_FLAGS = {
    "advanced_chunking": settings.enable_semantic_chunking,
    "hybrid_search": settings.enable_hybrid_search,
    "response_caching": settings.enable_caching,
    "model_routing": settings.enable_model_routing,
    "exact_pattern_search": settings.enable_exact_pattern_search,
    "keyword_search": settings.enable_keyword_search,
    "performance_monitoring": settings.enable_metrics,
    "query_logging": settings.enable_query_logging
}

print(f"🚀 Advanced features enabled: {[k for k, v in FEATURE_FLAGS.items() if v]}")