"""Unit tests for the Parent-Child Chunker."""

import src.ingestion.chunker as chunker_module
from src.ingestion.chunker import MIN_INTRO_LENGTH, ParentChildChunker, get_chunker


def test_split_by_headers():
    """Test if markdown is correctly split by H2/H3 headers."""
    chunker = ParentChildChunker()
    # Intro must be >= 300 chars to be kept
    intro = "This is a long introduction to ensure it passes the 300-character threshold filter. " * 5
    markdown = f"{intro}\n## Section 1\nContent 1\n### Section 2\nContent 2"

    sections = chunker.split_by_headers(markdown)

    assert len(sections) == 3
    assert sections[0][0] == "Introduction"
    assert "introduction" in sections[0][1].lower()
    assert sections[1][0] == "Section 1"
    assert "Content 1" in sections[1][1]


def test_chunk_document_parent_child_relationship():
    """Test if parent and child chunks are correctly created."""
    chunker = ParentChildChunker(child_chunk_size=50, min_child_length=10)
    # Parent text (header + content) must be >= 100
    # Child must be >= 150 (since we didn't override min_child_length in this test specifically to lower values)
    # Wait, the test uses min_child_length=10... but the DEFAULT is 150.
    # ParentChildChunker constructor: min_child_length=min_child_length (passed as 10)
    # So if the test passed 10, it should be fine.
    # Let me check the failure again: "assert 0 == 1".
    # Ah, ParentChildChunker.__init__: min_child_length=min_child_length (10)
    # BUT, parent_text length check in chunk_document uses self.min_parent_length (default 100).
    # markdown = "## Visa Rules\nThis is sentence one. This is sentence two. This is sentence three."
    # "Visa Rules\n\nThis is sentence one. This is sentence two. This is sentence three." -> 75 chars.
    # 75 < 100 -> dropped!
    content = (
        "This is a much longer sentence to ensure that our total parent chunk length exceeds the 100 character threshold. "
        * 2
    )
    markdown = f"## Visa Rules\n{content}"

    chunks = chunker.chunk_document(
        markdown_text=markdown, source_url="http://test.com", doc_id="doc_123", title="Test Doc"
    )

    # Check parent chunk
    parent_chunks = [c for c in chunks if c.metadata.is_parent]
    assert len(parent_chunks) == 1
    assert "Visa Rules" in parent_chunks[0].text

    # Check child chunks
    child_chunks = [c for c in chunks if not c.metadata.is_parent]
    assert len(child_chunks) > 0

    # Verify linking
    for child in child_chunks:
        assert child.metadata.parent_doc_id == "doc_123"
        assert child.metadata.section_header == "Visa Rules"


# ─── split_by_headers ────────────────────────────────────────────────────────


class TestSplitByHeaders:
    def test_short_intro_is_dropped(self):
        chunker = ParentChildChunker()
        # intro is too short (< MIN_INTRO_LENGTH = 300 chars)
        markdown = "Short intro.\n## Section\nContent here."
        sections = chunker.split_by_headers(markdown)
        headers = [s[0] for s in sections]
        assert "Introduction" not in headers
        assert "Section" in headers

    def test_no_headers_short_content_is_dropped(self):
        """Entire doc with no headers and short content → nothing kept."""
        chunker = ParentChildChunker()
        markdown = "Just a short line."
        sections = chunker.split_by_headers(markdown)
        assert sections == []

    def test_no_headers_long_content_kept_as_introduction(self):
        """Entire doc with no headers but long content → kept as Introduction."""
        chunker = ParentChildChunker()
        long_text = "x " * (MIN_INTRO_LENGTH // 2 + 10)  # > 300 chars
        sections = chunker.split_by_headers(long_text)
        assert len(sections) == 1
        assert sections[0][0] == "Introduction"

    def test_h3_header_creates_section(self):
        chunker = ParentChildChunker()
        markdown = "### Requirements\nYou need a passport."
        sections = chunker.split_by_headers(markdown)
        assert any(s[0] == "Requirements" for s in sections)

    def test_empty_section_content_not_appended(self):
        """A header with only whitespace content is not added as a section."""
        chunker = ParentChildChunker()
        markdown = "## Empty Section\n   \n## Real Section\nActual content here."
        sections = chunker.split_by_headers(markdown)
        assert not any(s[1].strip() == "" for s in sections)


# ─── _derive_context_label ───────────────────────────────────────────────────


class TestDeriveContextLabel:
    def test_generic_header_uses_title(self):
        chunker = ParentChildChunker()
        label = chunker._derive_context_label("Introduction", "Visa Requirements Guide")
        assert label == "Visa Requirements Guide"

    def test_non_generic_header_returned_as_is(self):
        chunker = ParentChildChunker()
        label = chunker._derive_context_label("Eligibility Criteria", "Visa Guide")
        assert label == "Eligibility Criteria"

    def test_overview_is_generic(self):
        chunker = ParentChildChunker()
        label = chunker._derive_context_label("Overview", "My Title")
        assert label == "My Title"

    def test_generic_header_no_title_returns_default(self):
        chunker = ParentChildChunker()
        label = chunker._derive_context_label("Introduction", "")
        assert label == "Overview"


# ─── split_into_sentences ────────────────────────────────────────────────────


class TestSplitIntoSentences:
    def test_empty_paragraphs_skipped(self):
        chunker = ParentChildChunker()
        text = "Para one.\n\n\n\nPara two."
        result = chunker.split_into_sentences(text, max_size=500)
        assert len(result) == 1  # Both fit in one chunk

    def test_small_paragraphs_combined(self):
        chunker = ParentChildChunker()
        text = "Para one.\n\nPara two.\n\nPara three."
        result = chunker.split_into_sentences(text, max_size=500)
        assert len(result) == 1
        combined = result[0]
        assert "Para one" in combined and "Para two" in combined

    def test_large_paragraph_split_by_sentence(self):
        chunker = ParentChildChunker()
        # Each sentence ~30 chars, max_size=60 → two sentences per chunk
        text = "First sentence here. Second sentence here. Third sentence here."
        result = chunker.split_into_sentences(text, max_size=40)
        assert len(result) >= 2

    def test_paragraph_exceeding_max_size_hard_split(self):
        """A paragraph with no sentence punctuation gets hard-split by character."""
        chunker = ParentChildChunker()
        # 200 chars with no sentence boundaries
        text = "a" * 200
        result = chunker.split_into_sentences(text, max_size=50)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 50

    def test_current_chunk_flushed_at_end(self):
        chunker = ParentChildChunker()
        text = "Short para."
        result = chunker.split_into_sentences(text, max_size=500)
        assert len(result) == 1
        assert "Short para" in result[0]


# ─── German statute structure ────────────────────────────────────────────────


# A statute page as the crawler delivers it: § number in the title, no H2/H3, and
# the rules addressed by Absatz.
STATUTE = """§ 18b Fachkräfte mit akademischer Ausbildung

(1) Einem Ausländer wird eine Aufenthaltserlaubnis zur Ausübung einer qualifizierten \
Beschäftigung erteilt, zu der seine Qualifikation ihn befähigt. § 18 Abs. 2 Nr. 4 ist \
nicht anzuwenden, vgl. § 19c Abs. 1 S. 2.

(2) Einem Ausländer wird abweichend von § 18 Abs. 2 Nr. 4 eine Blaue Karte EU erteilt, \
wenn er ein Gehalt in Höhe von mindestens 50 Prozent der jährlichen \
Beitragsbemessungsgrenze erhält und die Bundesagentur zugestimmt hat.

(3) Absatz 2 gilt entsprechend."""


class TestSplitByAbsatz:
    def test_one_unit_per_absatz(self):
        chunker = ParentChildChunker()
        units = chunker.split_by_absatz(STATUTE)
        assert [number for number, _ in units] == ["1", "2", "3"]

    def test_each_unit_keeps_its_marker(self):
        chunker = ParentChildChunker()
        for number, body in chunker.split_by_absatz(STATUTE):
            assert f"({number})" in body

    def test_absatz_bodies_do_not_leak_into_each_other(self):
        chunker = ParentChildChunker()
        units = dict(chunker.split_by_absatz(STATUTE))
        assert "Blaue Karte EU" in units["2"]
        assert "Blaue Karte EU" not in units["1"]

    def test_preamble_is_attached_to_the_first_absatz(self):
        """The § heading sits before "(1)" and must not be dropped."""
        chunker = ParentChildChunker()
        units = chunker.split_by_absatz(STATUTE)
        assert "§ 18b Fachkräfte mit akademischer Ausbildung" in units[0][1]

    def test_letter_suffixed_absatz(self):
        chunker = ParentChildChunker()
        text = "§ 19c Text\n\n(1) Erster Absatz.\n\n(2a) Eingefügter Absatz."
        assert [n for n, _ in chunker.split_by_absatz(text)] == ["1", "2a"]

    def test_prose_without_paragraph_sign_is_not_a_statute(self):
        chunker = ParentChildChunker()
        text = "Step (1) fill the form.\n\nStep (2) book an appointment."
        assert chunker.split_by_absatz(text) == []

    def test_single_marker_is_not_a_statute(self):
        chunker = ParentChildChunker()
        assert chunker.split_by_absatz("§ 4 Text\n\n(1) Nur ein Absatz.") == []


class TestPackAbsatzUnits:
    def test_short_absaetze_share_a_chunk_and_keep_their_markers(self):
        chunker = ParentChildChunker(child_chunk_size=512)
        packed = chunker.pack_absatz_units([("1", "(1) Kurz."), ("2", "(2) Auch kurz.")])
        assert len(packed) == 1
        label, body = packed[0]
        assert label == "Abs. 1-2"
        assert "(1)" in body and "(2)" in body

    def test_absaetze_are_not_merged_beyond_the_child_size(self):
        chunker = ParentChildChunker(child_chunk_size=60, min_child_length=10)
        packed = chunker.pack_absatz_units([("1", "(1) " + "a" * 50), ("2", "(2) " + "b" * 50)])
        assert [label for label, _ in packed] == ["Abs. 1", "Abs. 2"]

    def test_oversized_absatz_is_split_and_every_piece_keeps_its_label(self):
        chunker = ParentChildChunker(child_chunk_size=120)
        long_body = "(2) " + "Ein ordentlicher Satz über die Blaue Karte EU. " * 8
        packed = chunker.pack_absatz_units([("1", "(1) Kurz."), ("2", long_body)])
        pieces = [label for label, _ in packed if label == "Abs. 2"]
        assert len(pieces) > 1

    def test_short_tail_is_merged_backwards_so_it_survives_the_length_floor(self):
        chunker = ParentChildChunker(child_chunk_size=200, min_child_length=150)
        packed = chunker.pack_absatz_units([("1", "(1) " + "a" * 180), ("2", "(2) Absatz 1 gilt entsprechend.")])
        assert len(packed) == 1
        assert "(2) Absatz 1 gilt entsprechend." in packed[0][1]


class TestStatuteChunking:
    """The Absatz is a rule's legal address: § 18b Abs. 2 is the Blue Card, Abs. 1 the
    ordinary skilled worker permit. A child chunk must say which one it came from."""

    def _children(self, chunker):
        chunks = chunker.chunk_document(
            markdown_text=STATUTE,
            source_url="https://www.gesetze-im-internet.de/aufenthg_2004/__18b.html",
            doc_id="doc_18b",
            title="§ 18b AufenthG - Einzelnorm",
        )
        return [c for c in chunks if not c.metadata.is_parent]

    def test_every_child_names_its_absatz(self):
        children = self._children(ParentChildChunker(child_chunk_size=320, min_child_length=50))
        assert children
        for child in children:
            assert "| Abs. " in child.text.split("\n", 1)[0]

    def test_blue_card_rule_is_not_labelled_as_another_absatz(self):
        children = self._children(ParentChildChunker(child_chunk_size=320, min_child_length=50))
        carrying = [c for c in children if "Blaue Karte EU" in c.text]
        assert carrying
        for child in carrying:
            header = child.text.split("\n", 1)[0]
            assert "Abs. 2" in header
            assert "Abs. 1" not in header

    def test_prefix_does_not_repeat_the_title_as_section(self):
        """On a statute page the header is generic, so Section would echo Topic."""
        children = self._children(ParentChildChunker(child_chunk_size=320, min_child_length=50))
        assert all("Section:" not in c.text.split("\n", 1)[0] for c in children)

    def test_parent_section_opens_with_a_meaningful_heading(self):
        """Retrieval expands a hit into the parent, so "Introduction" is not enough."""
        chunker = ParentChildChunker(child_chunk_size=320, min_child_length=50)
        chunks = chunker.chunk_document(
            markdown_text=STATUTE,
            source_url="https://www.gesetze-im-internet.de/aufenthg_2004/__18b.html",
            doc_id="doc_18b",
            title="§ 18b AufenthG - Einzelnorm",
        )
        parent = next(c for c in chunks if c.metadata.is_parent)
        assert parent.text.startswith("§ 18b AufenthG - Einzelnorm")

    def test_children_link_back_to_their_parent_section(self):
        """Retrieval expands a child into its section, so the link must be explicit."""
        chunker = ParentChildChunker(child_chunk_size=320, min_child_length=50)
        chunks = chunker.chunk_document(
            markdown_text=STATUTE,
            source_url="https://www.gesetze-im-internet.de/aufenthg_2004/__18b.html",
            doc_id="doc_18b",
            title="§ 18b AufenthG - Einzelnorm",
        )
        parents = {c.metadata.chunk_id for c in chunks if c.metadata.is_parent}
        children = [c for c in chunks if not c.metadata.is_parent]
        assert children
        assert all(c.metadata.parent_chunk_id in parents for c in children)
        assert all(c.metadata.parent_chunk_id is None for c in chunks if c.metadata.is_parent)

    def test_ordinary_pages_get_no_absatz_label(self):
        chunker = ParentChildChunker(min_child_length=50)
        content = "You need a passport and proof of funds. " * 6
        chunks = chunker.chunk_document(
            markdown_text=f"## Requirements\n{content}",
            source_url="http://example.com",
            doc_id="doc_web",
            title="Visa Guide",
        )
        children = [c for c in chunks if not c.metadata.is_parent]
        assert children
        assert all("Abs. " not in c.text for c in children)


# ─── German legal abbreviations ──────────────────────────────────────────────


class TestSentenceSplittingOnLegalText:
    def test_citation_is_never_torn_apart(self):
        chunker = ParentChildChunker()
        para = (
            "Die Voraussetzung nach § 6 Abs. 1 S. 2 BeschV i. V. m. der Anlage ist erfüllt. "
            "Die Bundesagentur für Arbeit hat der Beschäftigung bereits zugestimmt."
        )
        result = chunker.split_into_sentences(para, max_size=100)
        assert len(result) == 2  # split at the real sentence end, not inside the citation
        assert "§ 6 Abs. 1 S. 2 BeschV i. V. m. der Anlage" in result[0]

    def test_no_piece_ends_on_a_dangling_abbreviation(self):
        chunker = ParentChildChunker()
        para = (
            "Nach § 18 Abs. 2 Nr. 4 ist die Zustimmung entbehrlich. "
            "Die Regelung gilt auch für Anträge nach § 19c Abs. 1 S. 2 AufenthG. "
            "Weitere Einzelheiten regelt die Beschäftigungsverordnung."
        )
        for piece in chunker.split_into_sentences(para, max_size=80):
            assert not piece.rstrip().endswith(("Abs.", "Nr.", "S.", "vgl.", "i.", "V.", "m."))

    def test_sentence_ending_in_a_year_still_splits(self):
        chunker = ParentChildChunker()
        result = chunker.split_into_sentences("Die Regel gilt seit 2024. Ein neuer Satz folgt.", max_size=30)
        assert len(result) == 2

    def test_oversized_enumeration_breaks_between_nummern(self):
        chunker = ParentChildChunker()
        para = (
            "(2) Die Blaue Karte EU wird erteilt, wenn eine der folgenden Voraussetzungen vorliegt: "
            "1. ein anerkannter ausländischer Hochschulabschluss von mindestens drei Jahren Dauer, "
            "2. eine Berufsqualifikation nach § 6 Abs. 1 S. 2 BeschV i. V. m. der Anlage, "
            "3. eine seit dem 1. März 2024 erworbene gleichwertige Qualifikation."
        )
        result = chunker.split_into_sentences(para, max_size=200)
        assert len(result) > 1
        # No piece may begin in the middle of a Nummer.
        for piece in result[1:]:
            assert piece.lstrip()[0].isdigit()

    def test_no_sentinel_leaks_into_output(self):
        chunker = ParentChildChunker()
        text = "Nach § 18 Abs. 2 Nr. 4 gilt dies. " * 20
        assert all("\x00" not in piece for piece in chunker.split_into_sentences(text, max_size=100))


# ─── chunk_document branches ────────────────────────────────────────────────


class TestChunkDocumentBranches:
    def test_oversized_parent_chunk_is_trimmed(self):
        from src.ingestion.chunker import MAX_CHUNK_LENGTH

        chunker = ParentChildChunker(min_child_length=10)
        # Create a section whose parent_text exceeds MAX_CHUNK_LENGTH
        huge_content = "word " * (MAX_CHUNK_LENGTH // 4)
        markdown = f"## Big Section\n{huge_content}"
        chunks = chunker.chunk_document(
            markdown_text=markdown,
            source_url="http://test.com",
            doc_id="doc_big",
            title="Big Doc",
        )
        parents = [c for c in chunks if c.metadata.is_parent]
        assert len(parents) == 1
        assert "[Truncated]" in parents[0].text

    def test_short_parent_section_skipped(self):
        chunker = ParentChildChunker(min_parent_length=200)
        # Section content that produces a parent_text < 200 chars
        markdown = "## Tiny\nHi."
        chunks = chunker.chunk_document(
            markdown_text=markdown,
            source_url="http://test.com",
            doc_id="doc_tiny",
            title="Tiny Doc",
        )
        assert chunks == []

    def test_short_child_chunk_skipped(self):
        chunker = ParentChildChunker(min_child_length=1000, min_parent_length=10)
        # Make sure parent passes but child is shorter than 1000
        content = "This is a short child." * 3  # ~66 chars — under 1000
        markdown = f"## Section\n{content}"
        chunks = chunker.chunk_document(
            markdown_text=markdown,
            source_url="http://test.com",
            doc_id="doc_x",
            title="X",
        )
        # Parent exists but no children
        assert any(c.metadata.is_parent for c in chunks)
        assert not any(not c.metadata.is_parent for c in chunks)

    def test_extract_urls_from_child(self):
        chunker = ParentChildChunker(min_child_length=10)
        url = "https://example.com/visa"
        content = ("word " * 30) + f" See {url} for details."
        markdown = f"## Links\n{content}"
        chunks = chunker.chunk_document(
            markdown_text=markdown,
            source_url="http://test.com",
            doc_id="doc_url",
            title="URL Doc",
        )
        children = [c for c in chunks if not c.metadata.is_parent]
        all_urls = [u for c in children for u in c.metadata.referenced_urls]
        assert url in all_urls

    def test_returns_empty_for_blank_markdown(self):
        chunker = ParentChildChunker()
        chunks = chunker.chunk_document(
            markdown_text="   ",
            source_url="http://test.com",
            doc_id="doc_blank",
            title="Blank",
        )
        assert chunks == []


# ─── clean_markdown ───────────────────────────────────────────────────────────


class TestCleanMarkdown:
    def test_removes_images(self):
        chunker = ParentChildChunker()
        text = "Before ![alt](http://img.com/a.png) After"
        result = chunker.clean_markdown(text)
        assert "![" not in result
        assert "Before" in result

    def test_removes_social_share_links(self):
        chunker = ParentChildChunker()
        text = "Content. [Teilen auf LinkedIn](https://linkedin.com/share) More."
        result = chunker.clean_markdown(text)
        assert "Teilen auf" not in result

    def test_removes_download_links(self):
        chunker = ParentChildChunker()
        text = "See [Download the form](https://example.com/form.pdf) here."
        result = chunker.clean_markdown(text)
        assert "Download" not in result

    def test_removes_gesetze_intra_law_navigation(self):
        """Prev/next Einzelnorm and table-of-contents nav links are stripped."""
        chunker = ParentChildChunker()
        text = (
            '[zurück](__18f.html "zur vorherigen Einzelnorm")\n\n'
            '[weiter](__18h.html "zur nachfolgenden Einzelnorm")\n\n'
            "[Nichtamtliches Inhaltsverzeichnis](index.html#BJNR195010004)\n\n"
            "# § 18g Blaue Karte EU\n\n(1) Einer Fachkraft ..."
        )
        result = chunker.clean_markdown(text)
        assert "__18f.html" not in result
        assert "__18h.html" not in result
        assert "Inhaltsverzeichnis" not in result
        assert "§ 18g Blaue Karte EU" in result  # statute text preserved

    def test_keeps_hyphens_in_german_compounds(self):
        """Hyphens carry meaning in German legal terms and must survive cleaning."""
        chunker = ParentChildChunker()
        text = "Das Fachkräfte-Einwanderungsgesetz gilt für Nicht-EU-Staatsangehörige."
        result = chunker.clean_markdown(text)
        assert "Fachkräfte-Einwanderungsgesetz" in result
        assert "Nicht-EU-Staatsangehörige" in result

    def test_still_removes_ui_symbols(self):
        chunker = ParentChildChunker()
        result = chunker.clean_markdown("Required ✔ documents ℹ️ here 📄")
        assert "✔" not in result and "📄" not in result
        assert "Required" in result

    def test_removes_the_gesetze_download_bar(self):
        """Emitted as a heading, so it became the section header of what followed."""
        chunker = ParentChildChunker()
        text = (
            "## Full text in format:   [HTML](englisch_aufenthg.html)  "
            '[PDF](englisch_aufenthg.pdf "pdf will be shown in separate tab")\n\n'
            "### § 18b Fachkräfte\n\n(1) Der Text bleibt."
        )
        result = chunker.clean_markdown(text)
        assert "Full text in format" not in result
        assert "englisch_aufenthg.pdf" not in result
        assert "§ 18b Fachkräfte" in result

    def test_keeps_ordinary_content_links(self):
        """The gesetze nav rule targets __NN.html / index.html only — other links survive."""
        chunker = ParentChildChunker()
        text = "See [the portal](https://example.com/page.html) for details."
        result = chunker.clean_markdown(text)
        assert "example.com/page.html" in result


# ─── get_chunker singleton ────────────────────────────────────────────────────


class TestGetChunker:
    def setup_method(self):
        chunker_module._chunker = None

    def teardown_method(self):
        chunker_module._chunker = None

    def test_returns_same_instance(self):
        a = get_chunker()
        b = get_chunker()
        assert a is b

    def test_returns_parent_child_chunker(self):
        assert isinstance(get_chunker(), ParentChildChunker)
