"""Tests for document splitters."""

import pytest

from pydantic_flow.rag.splitters import ChunkMetadata
from pydantic_flow.rag.splitters import DocumentChunk
from pydantic_flow.rag.splitters import MarkdownHeadingSplitter
from pydantic_flow.rag.splitters import SentenceSplitter
from pydantic_flow.rag.splitters import SimpleTokenCounter
from pydantic_flow.rag.splitters import SplitConfig
from pydantic_flow.rag.splitters import TokenSplitter


class TestSimpleTokenCounter:
    """Tests for SimpleTokenCounter."""

    def test_basic_counting(self) -> None:
        """Test basic token counting."""
        counter = SimpleTokenCounter(chars_per_token=4.0)

        assert counter.count("test") == 1
        assert counter.count("hello world") == 2
        assert counter.count("a" * 100) == 25

    def test_minimum_one_token(self) -> None:
        """Test that minimum is one token."""
        counter = SimpleTokenCounter()
        assert counter.count("a") == 1
        assert counter.count("") == 1


class TestTokenSplitter:
    """Tests for TokenSplitter."""

    def test_basic_splitting(self) -> None:
        """Test basic token splitting."""
        splitter = TokenSplitter()
        config = SplitConfig(max_tokens=10, overlap=2, min_chunk_chars=5)

        text = "word " * 100
        chunks = splitter.split(text, "test_doc", config)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.metadata.source_id == "test_doc" for c in chunks)

    def test_deterministic_ids(self) -> None:
        """Test that chunk IDs are deterministic."""
        splitter = TokenSplitter()
        config = SplitConfig(max_tokens=10, overlap=2)

        text = "test text for splitting"
        chunks1 = splitter.split(text, "doc1", config)
        chunks2 = splitter.split(text, "doc1", config)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=True):
            assert c1.id == c2.id

    def test_respects_min_chunk_size(self) -> None:
        """Test that minimum chunk size is respected."""
        splitter = TokenSplitter()
        config = SplitConfig(max_tokens=10, overlap=2, min_chunk_chars=50)

        text = "small"
        chunks = splitter.split(text, "test_doc", config)

        assert len(chunks) == 0

    def test_overlap_applied(self) -> None:
        """Test that overlap is correctly applied."""
        splitter = TokenSplitter()
        config = SplitConfig(max_tokens=5, overlap=1, min_chunk_chars=5)

        text = "one two three four five six seven eight nine ten"
        chunks = splitter.split(text, "test_doc", config)

        assert len(chunks) >= 2

        if len(chunks) >= 2:
            first_end = chunks[0].text[-10:]
            second_start = chunks[1].text[:10]
            assert any(word in second_start for word in first_end.split())

    def test_preserve_newlines(self) -> None:
        """Test newline preservation."""
        splitter = TokenSplitter()
        config = SplitConfig(
            max_chars=50, overlap=5, preserve_newlines=True, min_chunk_chars=10
        )

        text = "Line one.\nLine two.\n\nParagraph two.\nLine four."
        chunks = splitter.split(text, "test_doc", config)

        for chunk in chunks:
            if "\n" in chunk.text:
                assert not chunk.text.endswith("Line")


class TestSentenceSplitter:
    """Tests for SentenceSplitter."""

    def test_sentence_boundaries(self) -> None:
        """Test that sentences are not split mid-word."""
        splitter = SentenceSplitter()
        config = SplitConfig(max_chars=100, overlap=10, min_chunk_chars=10)

        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        chunks = splitter.split(text, "test_doc", config)

        for chunk in chunks:
            assert chunk.text.strip()
            assert not chunk.text.strip().endswith("First")
            assert not chunk.text.strip().endswith("Second")

    def test_paragraph_preservation(self) -> None:
        """Test that paragraphs are preserved."""
        splitter = SentenceSplitter()
        config = SplitConfig(
            max_chars=200, overlap=10, preserve_newlines=True, min_chunk_chars=10
        )

        text = (
            "Paragraph one sentence one. Paragraph one sentence two.\n\n"
            "Paragraph two sentence one."
        )
        chunks = splitter.split(text, "test_doc", config)

        assert len(chunks) > 0

    def test_overlap_by_sentences(self) -> None:
        """Test sentence overlap."""
        splitter = SentenceSplitter()
        config = SplitConfig(max_chars=60, overlap=30, min_chunk_chars=10)

        text = "A. B. C. D. E. F. G. H."
        chunks = splitter.split(text, "test_doc", config)

        if len(chunks) >= 2:
            assert len(chunks[0].text) <= 60 + 10
            assert len(chunks[1].text) <= 60 + 10

    def test_edge_case_no_punctuation(self) -> None:
        """Test text without sentence-ending punctuation."""
        splitter = SentenceSplitter()
        config = SplitConfig(max_chars=50, overlap=10, min_chunk_chars=5)

        text = "word " * 50
        chunks = splitter.split(text, "test_doc", config)

        assert len(chunks) > 0


class TestMarkdownHeadingSplitter:
    """Tests for MarkdownHeadingSplitter."""

    def test_heading_detection(self) -> None:
        """Test ATX heading detection."""
        splitter = MarkdownHeadingSplitter()
        config = SplitConfig(max_chars=1000, overlap=0, min_chunk_chars=5)

        text = """# Title

Content under title.

## Section

Content under section.

### Subsection

More content."""

        chunks = splitter.split(text, "test_doc", config)

        assert len(chunks) > 0
        assert any("Title" in c.metadata.heading_path for c in chunks)
        assert any("Section" in c.metadata.heading_path for c in chunks)

    def test_heading_path_hierarchy(self) -> None:
        """Test that heading paths maintain hierarchy."""
        splitter = MarkdownHeadingSplitter()
        config = SplitConfig(max_chars=1000, overlap=0, min_chunk_chars=5)

        text = """# Level 1

## Level 2

### Level 3

Content"""

        chunks = splitter.split(text, "test_doc", config)

        level3_chunks = [c for c in chunks if "Level 3" in c.metadata.heading_path]
        if level3_chunks:
            path = level3_chunks[0].metadata.heading_path
            assert "Level 1" in path
            assert "Level 2" in path
            assert "Level 3" in path

    def test_large_section_subdivision(self) -> None:
        """Test that large sections are subdivided."""
        splitter = MarkdownHeadingSplitter()
        config = SplitConfig(max_chars=100, overlap=20, min_chunk_chars=10)

        text = "# Heading\n\n" + ("Long content. " * 50)
        chunks = splitter.split(text, "test_doc", config)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 120

    def test_no_headings(self) -> None:
        """Test document without headings."""
        splitter = MarkdownHeadingSplitter()
        config = SplitConfig(max_chars=100, overlap=10, min_chunk_chars=10)

        text = "Just plain text without any headings."
        chunks = splitter.split(text, "test_doc", config)

        assert len(chunks) == 1
        assert chunks[0].metadata.heading_path == []

    def test_metadata_includes_heading_path(self) -> None:
        """Test that metadata includes heading path."""
        splitter = MarkdownHeadingSplitter()
        config = SplitConfig(max_chars=1000, overlap=0, min_chunk_chars=5)

        text = """# Main

## Sub

Content"""

        chunks = splitter.split(text, "test_doc", config)

        for chunk in chunks:
            assert isinstance(chunk.metadata.heading_path, list)


class TestSplitterEdgeCases:
    """Edge case tests across all splitters."""

    @pytest.mark.parametrize(
        "splitter_class",
        [TokenSplitter, SentenceSplitter, MarkdownHeadingSplitter],
    )
    def test_empty_text(self, splitter_class) -> None:
        """Test empty text handling."""
        splitter = splitter_class()
        config = SplitConfig(max_chars=100, overlap=10, min_chunk_chars=5)

        chunks = splitter.split("", "test_doc", config)
        assert chunks == []

    @pytest.mark.parametrize(
        "splitter_class",
        [TokenSplitter, SentenceSplitter, MarkdownHeadingSplitter],
    )
    def test_single_character(self, splitter_class) -> None:
        """Test single character text."""
        splitter = splitter_class()
        config = SplitConfig(max_chars=100, overlap=10, min_chunk_chars=1)

        chunks = splitter.split("a", "test_doc", config)
        assert len(chunks) <= 1

    @pytest.mark.parametrize(
        "splitter_class",
        [TokenSplitter, SentenceSplitter, MarkdownHeadingSplitter],
    )
    def test_metadata_consistency(self, splitter_class) -> None:
        """Test metadata consistency."""
        splitter = splitter_class()
        config = SplitConfig(max_chars=100, overlap=10, min_chunk_chars=5)

        text = "Test content for metadata consistency check."
        chunks = splitter.split(text, "test_doc", config)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata.source_id == "test_doc"
            assert chunk.metadata.chunk_index == i
            assert isinstance(chunk.metadata, ChunkMetadata)
