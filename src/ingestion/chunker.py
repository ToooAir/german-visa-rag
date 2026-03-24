"""
Advanced chunking strategy: Parent-Child (Small-to-Big) chunking.
- Large sections (H2/H3) are parent chunks for context
- Smaller semantic units are child chunks for precise retrieval
"""

from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime, timezone

from src.config import settings
from src.logger import logger
from src.models.chunk import Chunk, ChunkMetadata, VisaType, AuthorityLevel
from src.utils.hash_utils import compute_canonical_hash
from src.utils.text_utils import normalize_whitespace


# ── Minimum content thresholds ─────────────────────────────────────────────
MIN_CHILD_LENGTH  = 150   # chars — child chunks shorter than this are dropped
MIN_PARENT_LENGTH = 100   # chars — parent (section) chunks shorter than this are dropped

# ── "Introduction" section handling ────────────────────────────────────────
# Content before the first H2/H3 header is likely navigation or page-level
# boilerplate. We apply a stricter length threshold before keeping it.
MIN_INTRO_LENGTH  = 300   # chars — pre-header content must be at least this long


class ParentChildChunker:
    """
    Implements Small-to-Big retrieval strategy:

    1. Split markdown by H2/H3 headers (parent documents)
    2. Within each parent, split by sentences/paragraphs (child chunks)
    3. Embed child chunks for retrieval
    4. Pass parent + relevant child to LLM context
    """

    def __init__(
        self,
        child_chunk_size: int = None,
        child_chunk_overlap: int = None,
        parent_chunk_size: int = None,
        min_child_length: int = MIN_CHILD_LENGTH,
        min_parent_length: int = MIN_PARENT_LENGTH,
        min_intro_length: int = MIN_INTRO_LENGTH,
    ):
        self.child_chunk_size = child_chunk_size or settings.chunk_size
        self.child_chunk_overlap = child_chunk_overlap or settings.chunk_overlap
        self.parent_chunk_size = parent_chunk_size or settings.parent_chunk_size
        self.min_child_length = min_child_length
        self.min_parent_length = min_parent_length
        self.min_intro_length = min_intro_length

    def clean_markdown(self, text: str) -> str:
        """
        Remove unwanted markdown artifacts like images, social share links,
        download/print buttons, and navigation noise.
        """
        # Remove images: ![alt](url)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove common boilerplate links: [Download|Print...](url)
        text = re.sub(
            r'\[(?:Download|Print|View|Overview|Back to).*?\]\(.*?\)',
            '', text, flags=re.IGNORECASE
        )

        # Remove social sharing links (LinkedIn, Twitter/X, Facebook, WhatsApp, etc.)
        text = re.sub(
            r'\[(?:Teilen auf|Share on|Compartir en|分享到?)[^\]]*\]\([^\)]+\)\s*',
            '', text, flags=re.IGNORECASE
        )
        text = re.sub(
            r'\[(?:LinkedIn|Twitter|Facebook|WhatsApp|X \(vorher|Xing)[^\]]*\]\([^\)]+\)\s*',
            '', text, flags=re.IGNORECASE
        )

        # Remove markdown table rows that are pure separator or single-cell noise
        # e.g. "| | | | | |" — pipes with only whitespace between them
        text = re.sub(r'^\|(?:\s*\|)+\s*$', '', text, flags=re.MULTILINE)

        # Remove UI specific text fragments, symbols and metadata
        ui_noise_patterns = [
            r'Previous slide', r'Next slide', r'Slide \d+ of \d+',
            r'\[closed envelope E-Mail\]\(.*?\)', r'\[Hotline\]\(.*?\)', r'\[FAQ\]\(.*?\)',
            r'<desc>.*?</desc>',                     # SVG description labels
            r'©\s*.*?(?:\.com|\d{4})',               # Copyright credits
            r'[✔©✅ℹ️⚠️✅❌📊📄🔗📂📜📌📏\-]',     # UI symbols
            r'^.*?\]\(/en/working-in-germany/job-listings\?tx_solr.*$',  # job search leaks
            r'Translate it via your browser\.',
            r'Google Translate is a third-party provider\.',
            r'Find points of contact all over the world',
            r'\* \[Living in Germany\]\(.*?\) \* \[Housing & mobility\]\(.*?\)',  # Breadcrumbs
            r'\[WhatsApp\]\(WhatsApp:.*?\) \[Facebook\]\(http:.*?\) \[X\]\(https:.*?\)',
            r'\[Show more\]\(.*?\)', r'\[Share page\]\(.*?\)', r'\* ### Share page',
            r'(?:^|\s)\]\(https?://[^\s\)]+\)',      # Trailing broken link fragments
            r'^\*?\d{2}\.\d{2}\.\d{4}\*?$',         # Standalone date-only lines
            r'^\*?Pressemitteilung\*?$',             # Standalone Press Release tags
        ]
        for pattern in ui_noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove empty lines and normalize whitespace
        text = normalize_whitespace(text)
        return text

    def split_by_headers(self, markdown_text: str) -> List[Tuple[str, str]]:
        """
        Split markdown by H2/H3 headers.

        Pre-header content (before the first H2/H3) is only kept if it is
        substantive (>= MIN_INTRO_LENGTH chars), to avoid indexing navigation
        links and page-level boilerplate labelled as "Introduction".

        Returns:
            List of (header, content) tuples
        """
        pattern = r"^(#{2,3})\s+(.+?)$"

        lines = markdown_text.split("\n")
        sections = []
        current_header = None          # None = we haven't hit any header yet
        current_content: List[str] = []

        for line in lines:
            match = re.match(pattern, line)
            if match:
                # Flush the previous section
                if current_content:
                    content_text = "\n".join(current_content).strip()
                    if content_text:
                        if current_header is None:
                            # Pre-header intro — only keep if substantive
                            if len(content_text) >= self.min_intro_length:
                                sections.append(("Introduction", content_text))
                        else:
                            sections.append((current_header, content_text))
                    current_content = []

                current_header = match.group(2).strip()
            else:
                current_content.append(line)

        # Flush final section
        if current_content:
            content_text = "\n".join(current_content).strip()
            if content_text:
                if current_header is None:
                    # Entire document has no headers
                    if len(content_text) >= self.min_intro_length:
                        sections.append(("Introduction", content_text))
                else:
                    sections.append((current_header, content_text))

        logger.debug(f"Split markdown into {len(sections)} sections by headers")
        return sections

    def _derive_context_label(self, section_header: str, title: str) -> str:
        """
        Build a meaningful context label for child chunks.

        If the section is just "Introduction", fall back to the page title so
        the context prefix still provides useful retrieval signal.
        """
        generic_headers = {"introduction", "intro", "overview", "inhalt", "content"}
        if section_header.lower().strip() in generic_headers:
            # Use page title as the section label instead
            return title or "Overview"
        return section_header

    def split_into_sentences(self, text: str, max_size: int) -> List[str]:
        """
        Split text into sentences/paragraphs with size limit.

        Strategy:
        1. Split by paragraph (double newline)
        2. Split paragraphs by sentence if too large
        3. Recombine to reach max_size
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph itself is too large, split by sentences
            if len(para) > max_size:
                para_sentences = re.split(r'([.!?](?:\s+|$))', para)
                segments = []
                for i in range(0, len(para_sentences) - 1, 2):
                    segments.append(para_sentences[i] + para_sentences[i + 1])
                if len(para_sentences) % 2 == 1:
                    segments.append(para_sentences[-1])

                for sentence in segments:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    sentence_size = len(sentence)

                    if current_size + sentence_size > max_size and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_size = sentence_size
                    else:
                        current_chunk.append(sentence)
                        current_size += sentence_size + 1
            else:
                para_size = len(para)

                if current_size + para_size > max_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [para]
                    current_size = para_size
                else:
                    current_chunk.append(para)
                    current_size += para_size + 2

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if c.strip()]

    def chunk_document(
        self,
        markdown_text: str,
        source_url: str,
        doc_id: str,
        title: str,
        authority_level: AuthorityLevel = AuthorityLevel.THIRD_PARTY,
        visa_types: Optional[List[VisaType]] = None,
        language: str = "de",
        published_at: Optional[datetime] = None,
    ) -> List[Chunk]:
        """
        Create parent-child chunk structure.

        Args:
            markdown_text: Full document markdown
            source_url: Source URL
            doc_id: Document ID from state store
            title: Document title
            authority_level: Authority classification
            visa_types: Relevant visa types
            language: Document language
            published_at: Publication date

        Returns:
            List of Chunk objects (parent + children)
        """
        chunks = []
        fetched_at = datetime.now(timezone.utc)

        # Step 0: Clean markdown noise
        markdown_text = self.clean_markdown(markdown_text)

        # Step 1: Split by headers to get parent sections
        sections = self.split_by_headers(markdown_text)

        logger.info(
            "Chunking document",
            extra={
                "url": source_url,
                "sections": len(sections),
                "doc_id": doc_id,
            },
        )

        section_index = 0
        for section_header, section_content in sections:
            section_index += 1

            if not section_content.strip():
                continue

            # ── Parent chunk ─────────────────────────────────────────────
            parent_text = f"{section_header}\n\n{section_content}"

            # Drop trivially short parent sections (nav links, breadcrumbs, etc.)
            if len(parent_text.strip()) < self.min_parent_length:
                logger.debug(
                    f"Skipping short parent section '{section_header}' "
                    f"({len(parent_text)} chars) in {source_url}"
                )
                continue

            parent_hash = compute_canonical_hash(parent_text)
            parent_chunk_id = f"{doc_id}_section_{section_index}_parent"

            parent_chunk = Chunk(
                metadata=ChunkMetadata(
                    chunk_id=parent_chunk_id,
                    parent_doc_id=str(doc_id),
                    source_url=source_url,
                    source_title=title,
                    authority_level=authority_level,
                    visa_types=visa_types or [],
                    published_at=published_at,
                    fetched_at=fetched_at,
                    section_header=section_header,
                    is_parent=True,
                    parent_is_main_heading=True,
                    language=language,
                    text_hash=parent_hash,
                    referenced_urls=[],
                ),
                text=parent_text,
            )
            chunks.append(parent_chunk)

            # ── Child chunks ─────────────────────────────────────────────
            child_texts = self.split_into_sentences(
                section_content,
                max_size=self.child_chunk_size,
            )

            # Build a meaningful context label (avoid "Introduction" when generic)
            context_section_label = self._derive_context_label(section_header, title)
            display_title = title or source_url.split("/")[-1].replace("-", " ").title()

            child_index = 0
            for child_text in child_texts:
                child_index += 1

                child_text = child_text.strip()
                if not child_text or len(child_text) < self.min_child_length:
                    continue

                # Context Enhancement: "Topic: <page title> | Section: <meaningful header>"
                context_prefix = f"Topic: {display_title} | Section: {context_section_label}\n"
                enhanced_text = context_prefix + child_text

                child_hash = compute_canonical_hash(enhanced_text)
                child_chunk_id = f"{doc_id}_section_{section_index}_child_{child_index}"

                child_chunk = Chunk(
                    metadata=ChunkMetadata(
                        chunk_id=child_chunk_id,
                        parent_doc_id=str(doc_id),
                        source_url=source_url,
                        source_title=title,
                        authority_level=authority_level,
                        visa_types=visa_types or [],
                        published_at=published_at,
                        fetched_at=fetched_at,
                        section_header=section_header,
                        is_parent=False,
                        parent_is_main_heading=False,
                        language=language,
                        text_hash=child_hash,
                        referenced_urls=self._extract_urls(child_text),
                    ),
                    text=enhanced_text,
                )
                chunks.append(child_chunk)

        logger.info(
            "Chunking complete",
            extra={
                "total_chunks": len(chunks),
                "parent_chunks": sum(1 for c in chunks if c.metadata.is_parent),
                "child_chunks": sum(1 for c in chunks if not c.metadata.is_parent),
            },
        )

        return chunks

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from chunk text."""
        url_pattern = r'https?://[^\s\)]+'
        return list(set(re.findall(url_pattern, text)))


# Singleton instance
_chunker = None


def get_chunker() -> ParentChildChunker:
    """Get or create chunker singleton."""
    global _chunker
    if _chunker is None:
        _chunker = ParentChildChunker()
    return _chunker
