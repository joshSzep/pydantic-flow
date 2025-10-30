"""Token-based document splitter."""

import hashlib

from opentelemetry import trace

from pydantic_flow.rag.splitters.base import ChunkMetadata
from pydantic_flow.rag.splitters.base import DocumentChunk
from pydantic_flow.rag.splitters.base import SimpleTokenCounter
from pydantic_flow.rag.splitters.base import SplitConfig
from pydantic_flow.rag.splitters.base import TokenCounter

tracer = trace.get_tracer(__name__)


class TokenSplitter:
    """Token-approximate splitter with overlap.

    Splits documents by approximate token count with configurable overlap.
    Provides deterministic chunk IDs and respects size constraints.

    """

    def split(
        self,
        text: str,
        source_id: str,
        config: SplitConfig,
        token_counter: TokenCounter | None = None,
    ) -> list[DocumentChunk]:
        """Split document into token-sized chunks with overlap.

        Args:
            text: Document text to split.
            source_id: Identifier for the source document.
            config: Splitting configuration.
            token_counter: Optional token counter (uses SimpleTokenCounter if None).

        Returns:
            List of document chunks with metadata.

        """
        counter = token_counter or SimpleTokenCounter()

        with tracer.start_as_current_span("rag.split.token") as span:
            chunks = self._split_by_tokens(text, source_id, config, counter)

            if chunks:
                avg_size = sum(len(c.text) for c in chunks) / len(chunks)
                span.set_attribute("chunks", len(chunks))
                span.set_attribute("avg_size", int(avg_size))
                span.set_attribute("overlap", config.overlap)

            return chunks

    def _split_by_tokens(
        self,
        text: str,
        source_id: str,
        config: SplitConfig,
        counter: TokenCounter,
    ) -> list[DocumentChunk]:
        """Split text into token-based chunks."""
        if not text or len(text) < config.min_chunk_chars:
            return []

        max_size = config.max_tokens or int(config.max_chars / 4)
        overlap = config.overlap

        chunks: list[DocumentChunk] = []
        chunk_index = 0
        start = 0

        while start < len(text):
            end = len(text)
            current_text = text[start:end]

            tokens = counter.count(current_text)
            if tokens > max_size:
                target_chars = int(max_size * 4)
                end = start + target_chars

                while end > start and counter.count(text[start:end]) > max_size:
                    end = int(end * 0.9)

                if config.preserve_newlines:
                    last_newline = text.rfind("\n", start, end)
                    if last_newline > start + config.min_chunk_chars:
                        end = last_newline + 1

            chunk_text = text[start:end]

            if len(chunk_text) >= config.min_chunk_chars:
                chunk_id = self._generate_chunk_id(source_id, chunk_index, chunk_text)
                metadata = ChunkMetadata(
                    source_id=source_id,
                    chunk_index=chunk_index,
                    byte_start=start,
                    byte_end=end,
                    token_count=counter.count(chunk_text),
                )

                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break

            overlap_chars = int(overlap * 4) if config.max_tokens else overlap
            start = max(start + 1, end - overlap_chars)

        return chunks

    def _generate_chunk_id(self, source_id: str, index: int, text: str) -> str:
        """Generate deterministic chunk ID.

        Args:
            source_id: Source document ID.
            index: Chunk index.
            text: Chunk text content.

        Returns:
            Deterministic chunk identifier.

        """
        content = f"{source_id}:{index}:{text[:100]}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{source_id}_chunk_{index}_{hash_digest}"
