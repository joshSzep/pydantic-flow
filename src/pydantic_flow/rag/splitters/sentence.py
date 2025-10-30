"""Sentence-based document splitter."""

import hashlib
import re

from opentelemetry import trace

from pydantic_flow.rag.splitters.base import ChunkMetadata
from pydantic_flow.rag.splitters.base import DocumentChunk
from pydantic_flow.rag.splitters.base import SimpleTokenCounter
from pydantic_flow.rag.splitters.base import SplitConfig
from pydantic_flow.rag.splitters.base import TokenCounter

tracer = trace.get_tracer(__name__)

SENTENCE_ENDINGS = re.compile(r"([.!?]+[\s\n]+|[\n]{2,})")
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class SentenceSplitter:
    """Sentence-based splitter with paragraph awareness.

    Splits documents by sentences with punctuation heuristics.
    Preserves paragraph boundaries when possible.

    """

    def split(
        self,
        text: str,
        source_id: str,
        config: SplitConfig,
        token_counter: TokenCounter | None = None,
    ) -> list[DocumentChunk]:
        """Split document into sentence-based chunks with overlap.

        Args:
            text: Document text to split.
            source_id: Identifier for the source document.
            config: Splitting configuration.
            token_counter: Optional token counter (uses SimpleTokenCounter if None).

        Returns:
            List of document chunks with metadata.

        """
        counter = token_counter or SimpleTokenCounter()

        with tracer.start_as_current_span("rag.split.sentence") as span:
            chunks = self._split_by_sentences(text, source_id, config, counter)

            if chunks:
                span.set_attribute("chunks", len(chunks))
                span.set_attribute("avg_sentences", self._avg_sentence_count(chunks))

            return chunks

    def _split_by_sentences(
        self,
        text: str,
        source_id: str,
        config: SplitConfig,
        counter: TokenCounter,
    ) -> list[DocumentChunk]:
        """Split text into chunks at sentence boundaries."""
        if not text or len(text) < config.min_chunk_chars:
            return []

        paragraphs = PARAGRAPH_SPLIT.split(text) if config.preserve_newlines else [text]

        chunks: list[DocumentChunk] = []
        chunk_index = 0
        global_offset = 0

        for paragraph in paragraphs:
            if not paragraph.strip():
                global_offset += len(paragraph)
                continue

            para_chunks = self._split_paragraph(
                paragraph,
                source_id,
                config,
                counter,
                chunk_index,
                global_offset,
            )
            chunks.extend(para_chunks)
            chunk_index += len(para_chunks)
            global_offset += len(paragraph) + 2

        return chunks

    def _split_paragraph(  # noqa: PLR0913
        self,
        paragraph: str,
        source_id: str,
        config: SplitConfig,
        counter: TokenCounter,
        start_index: int,
        global_offset: int,
    ) -> list[DocumentChunk]:
        """Split a single paragraph into chunks."""
        sentences = self._split_sentences(paragraph)
        if not sentences:
            return []

        chunks: list[DocumentChunk] = []
        current_chunk: list[str] = []
        current_chars = 0
        local_offset = 0

        for sentence in sentences:
            sentence_chars = len(sentence)

            if (
                current_chars + sentence_chars > config.max_chars
                and current_chunk
                and current_chars >= config.min_chunk_chars
            ):
                chunk_text = "".join(current_chunk)
                chunk_start = global_offset + local_offset - current_chars

                chunk_id = self._generate_chunk_id(
                    source_id, start_index + len(chunks), chunk_text
                )
                metadata = ChunkMetadata(
                    source_id=source_id,
                    chunk_index=start_index + len(chunks),
                    byte_start=chunk_start,
                    byte_end=chunk_start + len(chunk_text),
                    token_count=counter.count(chunk_text),
                )

                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )

                if config.overlap > 0:
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk, config.overlap
                    )
                    current_chunk = overlap_sentences
                    current_chars = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_chars = 0

            current_chunk.append(sentence)
            current_chars += sentence_chars
            local_offset += sentence_chars

        if current_chunk and current_chars >= config.min_chunk_chars:
            chunk_text = "".join(current_chunk)
            chunk_start = global_offset + local_offset - current_chars

            chunk_id = self._generate_chunk_id(
                source_id, start_index + len(chunks), chunk_text
            )
            metadata = ChunkMetadata(
                source_id=source_id,
                chunk_index=start_index + len(chunks),
                byte_start=chunk_start,
                byte_end=chunk_start + len(chunk_text),
                token_count=counter.count(chunk_text),
            )

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using punctuation heuristics."""
        parts = SENTENCE_ENDINGS.split(text)
        sentences: list[str] = []

        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and SENTENCE_ENDINGS.match(parts[i + 1]):
                sentences.append(parts[i] + parts[i + 1])
                i += 2
            elif parts[i].strip():
                sentences.append(parts[i])
                i += 1
            else:
                i += 1

        return [s for s in sentences if s.strip()]

    def _get_overlap_sentences(
        self, sentences: list[str], overlap_chars: int
    ) -> list[str]:
        """Get last sentences that fit within overlap size."""
        overlap: list[str] = []
        chars = 0

        for sentence in reversed(sentences):
            if chars + len(sentence) > overlap_chars:
                break
            overlap.insert(0, sentence)
            chars += len(sentence)

        return overlap

    def _avg_sentence_count(self, chunks: list[DocumentChunk]) -> int:
        """Calculate average sentence count per chunk."""
        if not chunks:
            return 0

        total = sum(len(self._split_sentences(c.text)) for c in chunks)
        return int(total / len(chunks))

    def _generate_chunk_id(self, source_id: str, index: int, text: str) -> str:
        """Generate deterministic chunk ID."""
        content = f"{source_id}:{index}:{text[:100]}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{source_id}_chunk_{index}_{hash_digest}"
