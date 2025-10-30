"""Web loader."""

import hashlib
import re

import httpx

from pydantic_flow.rag.docs import Document
from pydantic_flow.rag.docs import Metadata
from pydantic_flow.rag.loaders.base import Loader


class WebLoader(Loader):
    """Load and extract text from web pages.

    Attributes:
        url: URL to fetch.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    """

    def __init__(
        self,
        url: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize web loader.

        Args:
            url: URL to fetch.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters.

        """
        self.url = url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def load(self) -> list[Document]:
        """Load documents from web.

        Returns:
            List of chunked documents.

        """
        async with httpx.AsyncClient() as client:
            response = await client.get(self.url, timeout=30.0)
            response.raise_for_status()
            html = response.text

        text = self._extract_text(html)
        chunks = self._chunk_text(text)

        documents = []
        for idx, chunk in enumerate(chunks):
            doc_id = self._generate_id(self.url, idx)
            metadata = Metadata(
                source=self.url,
                chunk_index=idx,
                total_chunks=len(chunks),
            )
            documents.append(
                Document(
                    id=doc_id,
                    content=chunk,
                    metadata=metadata,
                )
            )

        return documents

    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML.

        Args:
            html: HTML content.

        Returns:
            Extracted text.

        """
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk.

        Returns:
            List of text chunks.

        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _generate_id(self, source: str, chunk_index: int) -> str:
        """Generate unique ID for a document chunk.

        Args:
            source: Source identifier.
            chunk_index: Chunk index.

        Returns:
            Unique document ID.

        """
        content = f"{source}:{chunk_index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
