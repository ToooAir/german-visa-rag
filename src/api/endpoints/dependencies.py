from fastapi import Request
from src.rag.answer_generator import AnswerGenerator

def get_generator(request: Request) -> AnswerGenerator:
    """Dependency to get the initialized answer generator from app state."""
    if not hasattr(request.app.state, "answer_generator"):
        # Fallback for test clients that don't run lifespan
        from src.rag.hybrid_retriever import HybridRetriever
        from src.vector_db.qdrant_client_wrapper import get_qdrant_client
        qdrant = get_qdrant_client()
        retriever = HybridRetriever(qdrant_client=qdrant)
        request.app.state.answer_generator = AnswerGenerator(retriever=retriever)
        
    return request.app.state.answer_generator

def get_qdrant(request: Request):
    """Dependency to get the Qdrant client."""
    from src.vector_db.qdrant_client_wrapper import get_qdrant_client
    return get_qdrant_client()