"""Streaming parser for incremental JSON to Pydantic models.

This module provides tolerant parsing of partial JSON into structured
field updates, with graceful handling of malformed input.
"""

from collections.abc import AsyncIterator
import json
from typing import Any
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError

from pydantic_flow.streaming.events import NonFatalError
from pydantic_flow.streaming.events import PartialFields

T = TypeVar("T", bound=BaseModel)


class StreamingParser[T: BaseModel]:
    """Tolerant streaming parser for incremental JSON to Pydantic models.

    Handles partial JSON by attempting repair and extraction of complete fields.
    Emits PartialFields events as fields become available.
    """

    def __init__(self, model_type: type[T]) -> None:
        """Initialize the streaming parser.

        Args:
            model_type: The Pydantic model type to parse into.

        """
        self._model_type = model_type
        self._accumulated_text = ""
        self._extracted_fields: dict[str, Any] = {}

    async def astream_parse(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[PartialFields | NonFatalError | T]:
        """Parse incremental text into partial fields and final model.

        Args:
            text_stream: Async iterator of text chunks.

        Yields:
            PartialFields with incremental updates, NonFatalError for repair
            attempts, and final validated model at the end.

        """
        async for chunk in text_stream:
            self._accumulated_text += chunk

            # Try to extract complete fields from accumulated text
            new_fields = self._try_extract_fields()
            if new_fields:
                yield PartialFields(fields=new_fields)
                self._extracted_fields.update(new_fields)

        # At the end, try to construct final model
        try:
            final_model = self._finalize()
            yield final_model
        except ValidationError as e:
            # Emit non-fatal error for final validation failure
            yield NonFatalError(
                message=f"Final validation failed: {e}",
                recoverable=False,
            )
            # Re-raise to signal failure
            raise

    def _try_extract_fields(self) -> dict[str, Any]:
        """Attempt to extract complete fields from accumulated text.

        Returns:
            Dict of newly extracted fields, or empty dict if none found.

        """
        # Try parsing as-is
        parsed = self._try_parse_json(self._accumulated_text)
        if parsed:
            return self._diff_fields(parsed)

        # Try repair strategies
        repaired = self._repair_json(self._accumulated_text)
        if repaired:
            parsed = self._try_parse_json(repaired)
            if parsed:
                return self._diff_fields(parsed)

        return {}

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        """Attempt to parse JSON text.

        Args:
            text: JSON text to parse.

        Returns:
            Parsed dict or None if parsing failed.

        """
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return None

    def _repair_json(self, text: str) -> str | None:
        """Apply light repair strategies to partial JSON.

        Args:
            text: Partial JSON text.

        Returns:
            Repaired JSON text or None if no repair strategy worked.

        """
        text = text.strip()

        # Add missing closing braces
        if text.startswith("{"):
            open_braces = text.count("{")
            close_braces = text.count("}")
            if open_braces > close_braces:
                return text + "}" * (open_braces - close_braces)

        # Add missing closing bracket for arrays
        if "[" in text and text.count("[") > text.count("]"):
            return text + "]" * (text.count("[") - text.count("]"))

        return None

    def _diff_fields(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Extract newly discovered fields.

        Args:
            parsed: Newly parsed dict.

        Returns:
            Dict containing only new or changed fields.

        """
        new_fields = {}
        for key, value in parsed.items():
            is_new = key not in self._extracted_fields
            is_changed = not is_new and self._extracted_fields[key] != value
            if is_new or is_changed:
                new_fields[key] = value
        return new_fields

    def _finalize(self) -> T:
        """Construct final validated model from accumulated text.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValidationError: If final text cannot be parsed into valid model.

        """
        # Try final parse
        parsed = self._try_parse_json(self._accumulated_text)
        if not parsed:
            # Try repair one last time
            repaired = self._repair_json(self._accumulated_text)
            if repaired:
                parsed = self._try_parse_json(repaired)

        if not parsed:
            msg = f"Could not parse final JSON: {self._accumulated_text}"
            raise ValueError(msg)

        # Validate with Pydantic
        return self._model_type.model_validate(parsed)


async def parse_json_stream[T: BaseModel](
    text_stream: AsyncIterator[str],
    model_type: type[T],
) -> AsyncIterator[PartialFields | NonFatalError | T]:
    """Parse streaming JSON text into partial fields and final model.

    Args:
        text_stream: Async iterator of text chunks.
        model_type: The Pydantic model type to parse into.

    Yields:
        PartialFields with incremental updates, NonFatalError for repair
        attempts, and final validated model at the end.

    """
    parser = StreamingParser(model_type)
    async for item in parser.astream_parse(text_stream):
        yield item
