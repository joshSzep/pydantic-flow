"""Filesystem loader."""

import hashlib
from pathlib import Path

from pydantic_flow.rag.docs import Document
from pydantic_flow.rag.docs import Metadata
from pydantic_flow.rag.loaders.base import Loader


class FSLoader(Loader):
    """Load and chunk text files from filesystem.

    Attributes:
        path: Path to file or directory.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        extensions: File extensions to include (default: .txt, .md).

    """

    def __init__(
        self,
        path: str | Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        extensions: list[str] | None = None,
    ) -> None:
        """Initialize filesystem loader.

        Args:
            path: Path to file or directory.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters.
            extensions: File extensions to include.

        """
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extensions = extensions or [".txt", ".md"]

    async def load(self) -> list[Document]:
        """Load documents from filesystem.

        Returns:
            List of chunked documents.

        """
        documents = []

        if self.path.is_file():
            documents.extend(await self._load_file(self.path))
        elif self.path.is_dir():
            for file_path in self.path.rglob("*"):
                if file_path.is_file() and file_path.suffix in self.extensions:
                    documents.extend(await self._load_file(file_path))

        return documents

    async def _load_file(self, file_path: Path) -> list[Document]:
        """Load and chunk a single file.

        Args:
            file_path: Path to file.

        Returns:
            List of chunked documents from this file.

        """
        content = file_path.read_text(encoding="utf-8")
        chunks = self._chunk_text(content)

        documents = []
        for idx, chunk in enumerate(chunks):
            doc_id = self._generate_id(str(file_path), idx)
            metadata = Metadata(
                source=str(file_path),
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
