"""Markdown heading-based document splitter."""

import hashlib
import re

from opentelemetry import trace

from pydantic_flow.rag.splitters.base import ChunkMetadata
from pydantic_flow.rag.splitters.base import DocumentChunk
from pydantic_flow.rag.splitters.base import SimpleTokenCounter
from pydantic_flow.rag.splitters.base import SplitConfig
from pydantic_flow.rag.splitters.base import TokenCounter

tracer = trace.get_tracer(__name__)

ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class Section:
    """A markdown section with heading information."""

    def __init__(
        self,
        level: int,
        title: str,
        content: str,
        start_offset: int,
        parent_path: list[str],
    ) -> None:
        """Initialize section.

        Args:
            level: Heading level (1-6).
            title: Section title.
            content: Section content text.
            start_offset: Byte offset in source document.
            parent_path: List of parent heading titles.

        """
        self.level = level
        self.title = title
        self.content = content
        self.start_offset = start_offset
        self.parent_path = parent_path


class MarkdownHeadingSplitter:
    """Markdown heading-based splitter.

    Splits documents by ATX headings (# syntax) and preserves
    heading path metadata for context.

    """

    def split(
        self,
        text: str,
        source_id: str,
        config: SplitConfig,
        token_counter: TokenCounter | None = None,
    ) -> list[DocumentChunk]:
        """Split markdown document by headings.

        Args:
            text: Markdown document text to split.
            source_id: Identifier for the source document.
            config: Splitting configuration.
            token_counter: Optional token counter (uses SimpleTokenCounter if None).

        Returns:
            List of document chunks with heading metadata.

        """
        counter = token_counter or SimpleTokenCounter()

        with tracer.start_as_current_span("rag.split.markdown") as span:
            sections = self._parse_sections(text)
            chunks = self._sections_to_chunks(sections, source_id, config, counter)

            if chunks:
                span.set_attribute("sections", len(sections))
                span.set_attribute("chunks", len(chunks))

            return chunks

    def _parse_sections(self, text: str) -> list[Section]:
        """Parse markdown into sections based on headings."""
        sections: list[Section] = []
        matches = list(ATX_HEADING.finditer(text))

        if not matches:
            return [
                Section(
                    level=0,
                    title="",
                    content=text,
                    start_offset=0,
                    parent_path=[],
                )
            ]

        heading_stack: list[tuple[int, str]] = []

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.start()

            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            content = text[start:end]

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            parent_path = [h[1] for h in heading_stack]
            heading_stack.append((level, title))

            sections.append(
                Section(
                    level=level,
                    title=title,
                    content=content,
                    start_offset=start,
                    parent_path=parent_path,
                )
            )

        return sections

    def _sections_to_chunks(
        self,
        sections: list[Section],
        source_id: str,
        config: SplitConfig,
        counter: TokenCounter,
    ) -> list[DocumentChunk]:
        """Convert sections to chunks with optional subdivision."""
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._split_large_section(
                section, source_id, config, counter, chunk_index
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _split_large_section(
        self,
        section: Section,
        source_id: str,
        config: SplitConfig,
        counter: TokenCounter,
        start_index: int,
    ) -> list[DocumentChunk]:
        """Split large section into multiple chunks if needed."""
        content = section.content.strip()

        if not content or len(content) < config.min_chunk_chars:
            return []

        heading_path = [*section.parent_path, section.title] if section.title else []

        if len(content) <= config.max_chars:
            chunk_id = self._generate_chunk_id(source_id, start_index, content)
            metadata = ChunkMetadata(
                source_id=source_id,
                chunk_index=start_index,
                byte_start=section.start_offset,
                byte_end=section.start_offset + len(content),
                token_count=counter.count(content),
                heading_path=heading_path,
            )

            return [
                DocumentChunk(
                    id=chunk_id,
                    text=content,
                    metadata=metadata,
                )
            ]

        return self._subdivide_section(
            content,
            section.start_offset,
            heading_path,
            source_id,
            config,
            counter,
            start_index,
        )

    def _subdivide_section(  # noqa: PLR0913
        self,
        content: str,
        base_offset: int,
        heading_path: list[str],
        source_id: str,
        config: SplitConfig,
        counter: TokenCounter,
        start_index: int,
    ) -> list[DocumentChunk]:
        """Subdivide large section with overlap."""
        chunks: list[DocumentChunk] = []
        chunk_num = 0
        start = 0

        while start < len(content):
            end = min(start + config.max_chars, len(content))

            if end < len(content):
                newline_pos = content.rfind("\n", start, end)
                if newline_pos > start + config.min_chunk_chars:
                    end = newline_pos + 1

            chunk_text = content[start:end].strip()

            if len(chunk_text) >= config.min_chunk_chars:
                chunk_id = self._generate_chunk_id(
                    source_id, start_index + chunk_num, chunk_text
                )
                metadata = ChunkMetadata(
                    source_id=source_id,
                    chunk_index=start_index + chunk_num,
                    byte_start=base_offset + start,
                    byte_end=base_offset + end,
                    token_count=counter.count(chunk_text),
                    heading_path=heading_path,
                )

                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )
                chunk_num += 1

            if end >= len(content):
                break

            start = max(start + 1, end - config.overlap)

        return chunks

    def _generate_chunk_id(self, source_id: str, index: int, text: str) -> str:
        """Generate deterministic chunk ID."""
        content = f"{source_id}:{index}:{text[:100]}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{source_id}_chunk_{index}_{hash_digest}"
