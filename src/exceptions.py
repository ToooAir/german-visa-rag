"""Custom exceptions for the German Visa RAG application."""

class RAGException(Exception):
    """Base exception for RAG application."""
    pass

class IngestionError(RAGException):
    """Raised when document ingestion fails."""
    pass

class RetrievalError(RAGException):
    """Raised when vector database retrieval fails."""
    pass

class LLMGenerationError(RAGException):
    """Raised when LLM fails to generate an answer."""
    pass
