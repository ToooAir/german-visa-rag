"""
Data models for chunks, vectors, and metadata.
Defines the payload schema for Qdrant storage.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class AuthorityLevel(str, Enum):
    """Authority level of the document source."""
    OFFICIAL = "official"  # Bundesamtliche sources, official.de
    SEMI_OFFICIAL = "semi_official"  # Government agencies, universities
    THIRD_PARTY = "third_party"  # Blogs, news, forums


class VisaType(str, Enum):
    """Types of German visas and permits."""
    CHANCENKARTE = "chancenkarte"
    WORK_VISA = "work_visa"
    BLUE_CARD = "blue_card"
    IT_VISA = "it_visa"
    STUDENT_VISA = "student_visa"
    FAMILY_REUNION = "family_reunion"
    TOURIST_VISA = "tourist_visa"
    FREELANCE_VISA = "freelance_visa"
    ENTREPRENEUR_VISA = "entrepreneur_visa"
    GENERAL = "general"  # Cross-cutting regulations


class ChunkMetadata(BaseModel):
    """Metadata attached to each chunk."""
    
    # Identifiers
    chunk_id: str = Field(..., description="Unique chunk identifier (hash-based)")
    parent_doc_id: str = Field(..., description="Reference to parent document")
    
    # Source Information
    source_url: str = Field(..., description="Original URL of the document")
    source_title: Optional[str] = Field(default=None, description="Title of source page")
    
    # Authority & Classification
    authority_level: AuthorityLevel = Field(
        default=AuthorityLevel.THIRD_PARTY,
        description="Credibility level of source"
    )
    visa_types: List[VisaType] = Field(
        default_factory=list,
        description="Relevant visa categories"
    )
    
    # Temporal Information
    published_at: Optional[datetime] = Field(default=None, description="Original publication date")
    fetched_at: datetime = Field(default_factory=datetime.utcnow, description="Crawl timestamp")
    
    # Content Markers
    section_header: Optional[str] = Field(default=None, description="Heading context (H2/H3)")
    is_parent: bool = Field(default=False, description="Is this a parent/full document chunk?")
    parent_is_main_heading: bool = Field(
        default=False,
        description="Parent chunk represents main H2 section"
    )
    
    # Language & Processing
    language: str = Field(default="de", description="Language code (de/en/zh)")
    text_hash: str = Field(..., description="Canonical hash for deduplication")
    
    # Link tracking
    referenced_urls: List[str] = Field(
        default_factory=list,
        description="URLs referenced in chunk text"
    )


class Chunk(BaseModel):
    """Complete chunk with text and metadata."""
    
    metadata: ChunkMetadata
    text: str = Field(..., description="Main chunk text content")
    
    # Vectors will be computed on-the-fly during ingestion
    dense_vector: Optional[List[float]] = Field(
        default=None,
        description="Dense embedding vector (1536-dim for text-embedding-3-small)"
    )
    sparse_vector: Optional[Dict[int, float]] = Field(
        default=None,
        description="Sparse BM25 vector for hybrid search"
    )


class QdrantPayload(BaseModel):
    """Payload structure stored in Qdrant."""
    
    chunk_id: str
    parent_doc_id: str
    source_url: str
    source_title: Optional[str]
    authority_level: str  # Stored as string
    visa_types: List[str]  # Stored as list of strings
    published_at: Optional[str]  # ISO format
    fetched_at: str  # ISO format
    section_header: Optional[str]
    is_parent: bool
    language: str
    text_hash: str
    text: str  # Full text for retrieval
    referenced_urls: List[str]
    
    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "QdrantPayload":
        """Convert Chunk to QdrantPayload."""
        return cls(
            chunk_id=chunk.metadata.chunk_id,
            parent_doc_id=chunk.metadata.parent_doc_id,
            source_url=chunk.metadata.source_url,
            source_title=chunk.metadata.source_title,
            authority_level=chunk.metadata.authority_level.value,
            visa_types=[vt.value for vt in chunk.metadata.visa_types],
            published_at=chunk.metadata.published_at.isoformat() if chunk.metadata.published_at else None,
            fetched_at=chunk.metadata.fetched_at.isoformat(),
            section_header=chunk.metadata.section_header,
            is_parent=chunk.metadata.is_parent,
            language=chunk.metadata.language,
            text_hash=chunk.metadata.text_hash,
            text=chunk.text,
            referenced_urls=chunk.metadata.referenced_urls,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Qdrant upsert."""
        return self.model_dump(exclude_none=False)
