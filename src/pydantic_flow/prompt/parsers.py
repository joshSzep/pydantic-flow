"""Output parsers for structured data extraction."""

import json
from typing import Any

from pydantic import BaseModel

from pydantic_flow.prompt.types import OutputParser


class AsIsParser(OutputParser[str]):
    """Parser that returns text as-is without transformation."""

    async def parse(self, text: str) -> str:
        """Return text unchanged.

        Args:
            text: Raw text to return

        Returns:
            Unchanged text

        """
        return text


class JsonModelParser[T](OutputParser[T]):
    """Parser that extracts JSON and validates against a Pydantic model."""

    def __init__(self, model: type[T], *, strict: bool = True) -> None:
        """Initialize the JSON model parser.

        Args:
            model: Pydantic model class to validate against
            strict: Whether to use strict validation (default True)

        """
        self.model = model
        self.strict = strict

    async def parse(self, text: str) -> T:
        """Parse JSON text into validated model instance.

        Args:
            text: JSON text to parse

        Returns:
            Validated model instance

        Raises:
            json.JSONDecodeError: When text is not valid JSON
            pydantic.ValidationError: When JSON doesn't match model schema

        """
        obj: Any = json.loads(text)

        if isinstance(self.model, type) and issubclass(self.model, BaseModel):
            if self.strict:
                return self.model.model_validate(obj, strict=True)
            return self.model.model_validate(obj)

        if isinstance(obj, self.model):
            return obj
        msg = f"Cannot convert {type(obj).__name__} to {self.model}"
        raise TypeError(msg)


class DelimitedParser(OutputParser[dict[str, str]]):
    """Parser that splits text by delimiter into indexed dictionary."""

    def __init__(self, sep: str = "|") -> None:
        """Initialize the delimited parser.

        Args:
            sep: Delimiter to split on (default "|")

        """
        self.sep = sep

    async def parse(self, text: str) -> dict[str, str]:
        """Parse delimited text into indexed dictionary.

        Args:
            text: Delimited text to parse

        Returns:
            Dictionary mapping string indices to values

        """
        parts = [p.strip() for p in text.split(self.sep)]
        return {str(i): v for i, v in enumerate(parts)}
