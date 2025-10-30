"""PostgreSQL pgvector vector store."""

from typing import Any

import asyncpg

from pydantic_flow.rag.docs import Document
from pydantic_flow.rag.docs import Metadata
from pydantic_flow.rag.vectors.base import SearchResult
from pydantic_flow.rag.vectors.base import VectorStore


class PostgresPGVectorStore(VectorStore):
    """PostgreSQL vector store using pgvector extension.

    Attributes:
        pool: Connection pool.
        table_name: Name of the table storing vectors.
        dim: Embedding dimension.

    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "documents",
        dim: int = 1536,
    ) -> None:
        """Initialize PostgreSQL pgvector store.

        Args:
            connection_string: PostgreSQL connection string.
            table_name: Table name for storing documents.
            dim: Embedding dimension.

        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.dim = dim
        self.pool: asyncpg.Pool | None = None

    async def _ensure_pool(self) -> asyncpg.Pool:
        """Ensure connection pool is initialized.

        Returns:
            Connection pool.

        """
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.connection_string)
        return self.pool

    async def initialize(self) -> None:
        """Initialize the database schema.

        Creates the pgvector extension and table if they don't exist.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    embedding vector({self.dim}),
                    metadata JSONB
                )
                """
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx "
                f"ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)"
            )

    async def upsert(self, items: list[tuple[str, list[float], Document]]) -> None:
        """Upsert vectors and documents.

        Args:
            items: List of (id, vector, document) tuples.

        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            for doc_id, vector, document in items:
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    doc_id,
                    document.content,
                    vector,
                    document.metadata.model_dump_json(),
                )

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID.

        Args:
            ids: List of document IDs to delete.

        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self.table_name} WHERE id = ANY($1)", ids)

    async def query(
        self,
        vector: list[float],
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query for similar vectors.

        Args:
            vector: Query embedding vector.
            k: Number of results to return.
            filter: Optional metadata filter (JSONB queries).

        Returns:
            List of search results ordered by similarity.

        """
        pool = await self._ensure_pool()
        filter_clause = ""
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(f"metadata->>{key} = '{value}'")
            if conditions:
                filter_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT id, content, metadata, 1 - (embedding <=> $1) as score
            FROM {self.table_name}
            {filter_clause}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, vector, k)

        results = []
        for row in rows:
            metadata = Metadata.model_validate_json(row["metadata"])
            document = Document(
                id=row["id"],
                content=row["content"],
                metadata=metadata,
            )
            results.append(
                SearchResult(
                    id=row["id"],
                    document=document,
                    score=float(row["score"]),
                    metadata={},
                )
            )

        return results

    def embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension.

        """
        return self.dim

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
