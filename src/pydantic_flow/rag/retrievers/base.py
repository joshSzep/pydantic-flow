"""Retriever abstract base class."""

from abc import ABC
from abc import abstractmethod

from pydantic_flow.rag.docs import Document


class Retriever(ABC):
    """Abstract base class for retrievers.

    Implementations must provide retrieve() method.
    """

    @abstractmethod
    async def retrieve(self, query: str, k: int) -> list[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query string.
            k: Number of documents to retrieve.

        Returns:
            List of retrieved documents.

        """
        ...
