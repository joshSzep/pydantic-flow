"""Loader abstract base class."""

from abc import ABC
from abc import abstractmethod

from pydantic_flow.rag.docs import Document


class Loader(ABC):
    """Abstract base class for document loaders.

    Implementations must provide load() method.
    """

    @abstractmethod
    async def load(self) -> list[Document]:
        """Load documents from source.

        Returns:
            List of loaded documents.

        """
        ...
