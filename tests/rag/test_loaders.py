"""Tests for RAG loaders."""

from pathlib import Path
import tempfile

import pytest

from pydantic_flow.rag.loaders.fs import FSLoader


@pytest.mark.asyncio
async def test_fs_loader_single_file():
    """Test FSLoader with single file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test file with some content.")
        temp_path = Path(f.name)

    try:
        loader = FSLoader(path=temp_path, chunk_size=20, chunk_overlap=5)
        documents = await loader.load()

        assert len(documents) > 0
        assert all(doc.metadata.source == str(temp_path) for doc in documents)
        assert documents[0].metadata.chunk_index == 0
        assert documents[0].metadata.total_chunks == len(documents)
    finally:
        temp_path.unlink()


@pytest.mark.asyncio
async def test_fs_loader_directory():
    """Test FSLoader with directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        (temp_path / "file1.txt").write_text("Content of file 1")
        (temp_path / "file2.md").write_text("Content of file 2")
        (temp_path / "file3.py").write_text("Content of file 3")

        loader = FSLoader(path=temp_path, extensions=[".txt", ".md"])
        documents = await loader.load()

        sources = {doc.metadata.source for doc in documents}
        assert str(temp_path / "file1.txt") in sources
        assert str(temp_path / "file2.md") in sources
        assert str(temp_path / "file3.py") not in sources


@pytest.mark.asyncio
async def test_fs_loader_chunking():
    """Test FSLoader chunking behavior."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        content = "A" * 100
        f.write(content)
        temp_path = Path(f.name)

    try:
        loader = FSLoader(path=temp_path, chunk_size=30, chunk_overlap=10)
        documents = await loader.load()

        assert len(documents) > 1
        for i, doc in enumerate(documents):
            assert doc.metadata.chunk_index == i
            assert len(doc.content) <= 30
    finally:
        temp_path.unlink()


@pytest.mark.asyncio
async def test_web_loader():
    """Test WebLoader."""
    from unittest.mock import AsyncMock
    from unittest.mock import MagicMock
    from unittest.mock import patch

    # Mock httpx response
    mock_response = MagicMock()
    mock_response.text = (
        "Example Domain\n\nThis domain is for use in illustrative examples."
    )
    mock_response.raise_for_status = MagicMock()

    mock_get = AsyncMock(return_value=mock_response)

    mock_context = AsyncMock()
    mock_context.__aenter__.return_value.get = mock_get

    with patch("pydantic_flow.rag.loaders.web.httpx.AsyncClient") as mock_client:
        mock_client.return_value = mock_context

        from pydantic_flow.rag.loaders.web import WebLoader

        loader = WebLoader(url="https://example.com", chunk_size=500)
        documents = await loader.load()

        assert len(documents) > 0
        assert all(doc.metadata.source == "https://example.com" for doc in documents)
        assert documents[0].metadata.chunk_index == 0
