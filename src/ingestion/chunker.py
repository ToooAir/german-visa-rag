"""
Advanced chunking strategy: Parent-Child (Small-to-Big) chunking.
- Large sections (H2/H3) are parent chunks for context
- Smaller semantic units are child chunks for precise retrieval
"""

import re
from datetime import datetime, timezone
from typing import Optional

from src.config import settings
from src.logger import logger
from src.models.chunk import AuthorityLevel, Chunk, ChunkMetadata, VisaType
from src.utils.hash_utils import compute_canonical_hash
from src.utils.text_utils import normalize_whitespace

# ── Minimum content thresholds ─────────────────────────────────────────────
MIN_CHILD_LENGTH = 150  # chars — child chunks shorter than this are dropped
MIN_PARENT_LENGTH = 100  # chars — parent (section) chunks shorter than this are dropped

# Content before the first H2/H3 header is likely navigation or page-level
# boilerplate. We apply a stricter length threshold before keeping it.
MIN_INTRO_LENGTH = 300  # chars — pre-header content must be at least this long

# ── Safety limits (to avoid OpenAI 8192 token limit) ──────────────────────
# 8192 tokens is roughly 30,000 - 40,000 characters for western languages.
# We set a safe upper bound of 20,000 chars for any single chunk.
MAX_CHUNK_LENGTH = 20000

# ── German statute structure ──────────────────────────────────────────────
# A gesetze-im-internet.de "Einzelnorm" page carries no H2/H3 headings, so header
# splitting leaves the whole § as one flat section. The legal address of a rule is
# its Absatz -- § 18b Abs. 2 grants the Blue Card, Abs. 1 the ordinary skilled
# worker permit -- so a child chunk must never straddle an Absatz boundary nor be
# cut loose from its number.
ABSATZ_MARKER = re.compile(r"(?:^|\n)[ \t]*\((\d{1,2}[a-z]?)\)\s+", re.MULTILINE)

# Below this many markers the text is prose that happens to use "(1)", not a statute.
MIN_ABSATZ_MARKERS = 2

# Boundary between two Nummern of the same Absatz ("..., 2. ein anerkannter ..."). A
# Nummer is part of a rule's address (§ 18b Abs. 2 Nr. 3), so an oversized Absatz is
# cut here rather than mid-Nummer. The list stem keeps its first item: the colon that
# introduces the list is deliberately not a boundary.
NUMMER_MARKER = re.compile(r"(?<=[,;])\s+(?=\d{1,2}\.\s)")

# Periods that do not end a sentence in German legal prose. Citation abbreviations
# ("Abs.", "Nr."), any single letter ("i. V. m.", "S. 2"), and short numbers used
# for enumeration and ordinals ("3. eine Berufsqualifikation", "1. März"). Years
# are four digits and stay excluded, so a sentence ending in one still splits.
_LEGAL_ABBREVIATIONS = (
    "Abs",
    "Abschn",
    "Anl",
    "Art",
    "Aufl",
    "bspw",
    "Buchst",
    "bzw",
    "evtl",
    "ff",
    "ggf",
    "Hrsg",
    "inkl",
    "lit",
    "Nr",
    "Nrn",
    "Rn",
    "Satz",
    "sog",
    "vgl",
    "Ziff",
)
NON_TERMINAL_DOT = re.compile(
    r"(\b(?:" + "|".join(_LEGAL_ABBREVIATIONS) + r")|\b[A-Za-zÄÖÜäöüß]|(?<!\d)\d{1,2})\.(?=\s|$)"
)

# Stands in for a protected period while sentence splitting runs. Same width as the
# period it replaces, so every length check downstream stays accurate.
_DOT_SENTINEL = "\x00"


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
        min_child_length: int = MIN_CHILD_LENGTH,
        min_parent_length: int = MIN_PARENT_LENGTH,
        min_intro_length: int = MIN_INTRO_LENGTH,
    ):
        self.child_chunk_size = child_chunk_size or settings.chunk_size
        self.min_child_length = min_child_length
        self.min_parent_length = min_parent_length
        self.min_intro_length = min_intro_length

    def clean_markdown(self, text: str) -> str:
        """
        Remove unwanted markdown artifacts like images, social share links,
        download/print buttons, and navigation noise.
        """
        # Remove images: ![alt](url)
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove common boilerplate links: [Download|Print...](url)
        text = re.sub(r"\[(?:Download|Print|View|Overview|Back to).*?\]\(.*?\)", "", text, flags=re.IGNORECASE)

        # Remove gesetze-im-internet.de intra-law navigation: prev/next Einzelnorm
        # ([zurück](__18f.html), [weiter](__18h.html)) and the table-of-contents
        # link ([Nichtamtliches Inhaltsverzeichnis](index.html#...)). These target
        # sibling-norm or index pages and are pure navigation, not statute text.
        text = re.sub(r"\[[^\]]*\]\((?:__\w+\.html|index\.html)[^)]*\)\s*", "", text, flags=re.IGNORECASE)

        # Remove social sharing links (LinkedIn, Twitter/X, Facebook, WhatsApp, etc.)
        text = re.sub(
            r"\[(?:Teilen auf|Share on|Compartir en|分享到?)[^\]]*\]\([^\)]+\)\s*", "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\[(?:LinkedIn|Twitter|Facebook|WhatsApp|X \(vorher|Xing)[^\]]*\]\([^\)]+\)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove markdown table rows that are pure separator or single-cell noise
        # e.g. "| | | | | |" — pipes with only whitespace between them
        text = re.sub(r"^\|(?:\s*\|)+\s*$", "", text, flags=re.MULTILINE)

        # Remove UI specific text fragments, symbols and metadata
        ui_noise_patterns = [
            r"Previous slide",
            r"Next slide",
            r"Slide \d+ of \d+",
            r"\[closed envelope E-Mail\]\(.*?\)",
            r"\[Hotline\]\(.*?\)",
            r"\[FAQ\]\(.*?\)",
            r"<desc>.*?</desc>",  # SVG description labels
            r"©\s*.*?(?:\.com|\d{4})",  # Copyright credits
            # UI symbols. Hyphens are deliberately absent: German compounds carry
            # them ("Fachkräfte-Einwanderungsgesetz", "Nicht-EU-Staatsangehörige")
            # and stripping them breaks both lexical search and the term itself.
            r"[✔©✅ℹ️⚠️❌📊📄🔗📂📜📌📏]",
            r"^.*?\]\(/en/working-in-germany/job-listings\?tx_solr.*$",  # job search leaks
            r"Translate it via your browser\.",
            r"Google Translate is a third-party provider\.",
            r"Find points of contact all over the world",
            r"\* \[Living in Germany\]\(.*?\) \* \[Housing & mobility\]\(.*?\)",  # Breadcrumbs
            r"\[WhatsApp\]\(WhatsApp:.*?\) \[Facebook\]\(http:.*?\) \[X\]\(https:.*?\)",
            r"\[Show more\]\(.*?\)",
            r"\[Share page\]\(.*?\)",
            r"\* ### Share page",
            r"(?:^|\s)\]\(https?://[^\s\)]+\)",  # Trailing broken link fragments
            r"^\*?\d{2}\.\d{2}\.\d{4}\*?$",  # Standalone date-only lines
            r"^\*?Pressemitteilung\*?$",  # Standalone Press Release tags
        ]
        for pattern in ui_noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove empty lines and normalize whitespace
        text = normalize_whitespace(text)
        return text

    def split_by_headers(self, markdown_text: str) -> list[tuple[str, str]]:
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
        sections: list[tuple[str, str]] = []
        current_header = None  # None = we haven't hit any header yet
        current_content: list[str] = []

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

        logger.debug("Split markdown into %d sections by headers", len(sections))
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

    def split_into_sentences(self, text: str, max_size: int) -> list[str]:
        """
        Split text into sentences/paragraphs with size limit.

        Strategy:
        1. Split by paragraph (double newline)
        2. Split paragraphs by sentence if too large
        3. Recombine to reach max_size
        """
        paragraphs = [part for para in text.split("\n\n") for part in self._split_enumerations(para, max_size)]
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph itself is too large, split by sentences
            if len(para) > max_size:
                # 1. Try splitting by sentence punctuation first. Periods that
                # belong to a citation or an enumerator are masked so they cannot
                # be mistaken for a sentence end; the mask is lifted on return.
                protected = NON_TERMINAL_DOT.sub(lambda m: m.group(1) + _DOT_SENTINEL, para)
                para_sentences = re.split(r"([.!?](?:\s+|$))", protected)
                segments = []
                for i in range(0, len(para_sentences) - 1, 2):
                    segments.append(para_sentences[i] + para_sentences[i + 1])
                if len(para_sentences) % 2 == 1:
                    segments.append(para_sentences[-1])

                # 2. Hard Fallback: If any segment is still too large (no punctuation),
                # split by character length
                final_segments = []
                for seg in segments:
                    if len(seg) > max_size:
                        # Force split into max_size chunks
                        for j in range(0, len(seg), max_size):
                            final_segments.append(seg[j : j + max_size])
                    else:
                        final_segments.append(seg)

                for sentence in final_segments:
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

        return [c.replace(_DOT_SENTINEL, ".") for c in chunks if c.strip()]

    @staticmethod
    def _split_enumerations(paragraph: str, max_size: int) -> list[str]:
        """Break an oversized paragraph at its statute enumeration boundaries.

        Only applied when the paragraph does not fit a child chunk; the packer
        recombines the parts that do fit, so the boundary lands between two Nummern
        instead of inside one. A part separated from its list stem still carries its
        own "N." marker and the Absatz label, which is as much address as can be kept
        without duplicating the stem into every part.
        """
        if len(paragraph) <= max_size:
            return [paragraph]
        return [part for part in NUMMER_MARKER.split(paragraph) if part.strip()]

    def split_by_absatz(self, text: str) -> list[tuple[str, str]]:
        """Split German statute text into one (number, text) unit per Absatz.

        Returns an empty list when the text is not a statute, so ordinary pages
        fall through to the generic paragraph splitter. Any preamble before the
        first marker -- on an Einzelnorm page that is the § heading -- is kept and
        attached to the first Absatz rather than dropped.
        """
        if "§" not in text:
            return []

        matches = list(ABSATZ_MARKER.finditer(text))
        if len(matches) < MIN_ABSATZ_MARKERS:
            return []

        units: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[match.start() : end].strip()
            if not body:
                continue
            if not units and preamble:
                body = f"{preamble}\n\n{body}"
            units.append((match.group(1), body))

        return units

    def pack_absatz_units(self, units: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Group whole Absätze into child-sized units without straddling one.

        Consecutive short Absätze share a chunk and each keeps its own "(n)" marker
        inline. An Absatz longer than the child size is split, and every piece is
        labelled with that Absatz. A trailing group under min_child_length is merged
        backwards, so a short provision ("(3) Absatz 2 gilt entsprechend.") is not
        dropped by the length floor later on.
        """
        packed: list[tuple[list[str], str]] = []
        buffer: list[str] = []
        numbers: list[str] = []

        def flush() -> None:
            if buffer:
                packed.append((list(numbers), "\n\n".join(buffer)))
                buffer.clear()
                numbers.clear()

        for number, body in units:
            if len(body) > self.child_chunk_size:
                flush()
                for piece in self.split_into_sentences(body, max_size=self.child_chunk_size):
                    packed.append(([number], piece))
                continue

            if buffer and len("\n\n".join(buffer)) + len(body) + 2 > self.child_chunk_size:
                flush()
            buffer.append(body)
            numbers.append(number)

        flush()

        if len(packed) > 1 and len(packed[-1][1]) < self.min_child_length:
            tail_numbers, tail_body = packed.pop()
            prev_numbers, prev_body = packed[-1]
            packed[-1] = (prev_numbers + tail_numbers, f"{prev_body}\n\n{tail_body}")

        return [(self._absatz_label(nums), body) for nums, body in packed]

    @staticmethod
    def _absatz_label(numbers: list[str]) -> str:
        """Render the Absätze a chunk covers, e.g. "Abs. 2" or "Abs. 2-3"."""
        unique = list(dict.fromkeys(numbers))
        if len(unique) == 1:
            return f"Abs. {unique[0]}"
        return f"Abs. {unique[0]}-{unique[-1]}"

    def chunk_document(
        self,
        markdown_text: str,
        source_url: str,
        doc_id: str,
        title: str,
        authority_level: AuthorityLevel = AuthorityLevel.THIRD_PARTY,
        visa_types: Optional[list[VisaType]] = None,
        language: str = "de",
        published_at: Optional[datetime] = None,
    ) -> list[Chunk]:
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

            # Safety check: Trim parent if it's monstrously large
            if len(parent_text) > MAX_CHUNK_LENGTH:
                logger.warning("Trimming oversized parent chunk (%d chars) for %s", len(parent_text), source_url)
                parent_text = parent_text[:MAX_CHUNK_LENGTH] + "... [Truncated]"

            # Drop trivially short parent sections (nav links, breadcrumbs, etc.)
            if len(parent_text.strip()) < self.min_parent_length:
                logger.debug(
                    "Skipping short parent section '%s' (%d chars) in %s",
                    section_header,
                    len(parent_text),
                    source_url,
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
            # Statute sections are cut along their Absätze; everything else by
            # paragraph. child_units is (absatz_label or "", text).
            absatz_units = self.split_by_absatz(section_content)
            if absatz_units:
                child_units = self.pack_absatz_units(absatz_units)
            else:
                child_units = [
                    ("", text)
                    for text in self.split_into_sentences(
                        section_content,
                        max_size=self.child_chunk_size,
                    )
                ]

            # Build a meaningful context label (avoid "Introduction" when generic)
            context_section_label = self._derive_context_label(section_header, title)
            display_title = title or source_url.split("/")[-1].replace("-", " ").title()

            child_index = 0
            for absatz_label, child_text in child_units:
                child_index += 1

                child_text = child_text.strip()
                if not child_text or len(child_text) < self.min_child_length:
                    continue

                # Context Enhancement: "Topic: <page title> | Section: <meaningful
                # header>", plus "| Abs. <n>" so a retrieved statute fragment still
                # states which Absatz it came from.
                prefix_parts = [f"Topic: {display_title}"]
                if context_section_label != display_title:
                    prefix_parts.append(f"Section: {context_section_label}")
                if absatz_label:
                    prefix_parts.append(absatz_label)
                enhanced_text = " | ".join(prefix_parts) + "\n" + child_text

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

    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from chunk text."""
        url_pattern = r"https?://[^\s\)]+"
        return list(set(re.findall(url_pattern, text)))


# Singleton instance
_chunker = None


def get_chunker() -> ParentChildChunker:
    """Get or create chunker singleton."""
    global _chunker
    if _chunker is None:
        _chunker = ParentChildChunker()
    return _chunker
