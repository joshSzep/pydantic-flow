"""Data redaction for PII and sensitive information in checkpoints.

Provides pattern-based and field-level redaction for compliance.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Any

from pydantic import BaseModel
from pydantic import Field

# Constants
PARTIAL_REDACTION_MIN_LENGTH = 7


class PIIPattern(str, Enum):
    """Common PII patterns for redaction.

    Attributes:
        SSN: US Social Security Numbers (###-##-####).
        CREDIT_CARD: Credit card numbers (16 digits).
        EMAIL: Email addresses.
        PHONE: Phone numbers (various formats).
        IP_ADDRESS: IPv4 addresses.
        API_KEY: Common API key patterns.

    """

    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"


class RedactionStrategy(str, Enum):
    """Strategy for redacting sensitive data.

    Attributes:
        MASK: Replace with asterisks (e.g., "***-**-1234").
        HASH: Replace with SHA256 hash (deterministic, for matching).
        REMOVE: Remove entirely from data.
        PARTIAL: Show first/last chars (e.g., "***-**-1234").

    """

    MASK = "mask"
    HASH = "hash"
    REMOVE = "remove"
    PARTIAL = "partial"


class RedactionPolicy(BaseModel):
    """Policy for data redaction.

    Attributes:
        enabled: Whether redaction is enabled.
        patterns: PII patterns to redact.
        strategy: How to redact matches.
        field_names: Specific field names to redact.
        custom_patterns: Custom regex patterns to redact.
        preserve_length: Keep original string length when masking.

    Example:
        >>> policy = RedactionPolicy(
        ...     enabled=True,
        ...     patterns=[PIIPattern.EMAIL, PIIPattern.SSN],
        ...     strategy=RedactionStrategy.MASK,
        ...     field_names=["password", "api_key"],
        ... )

    """

    enabled: bool = True
    patterns: list[PIIPattern] = Field(default_factory=list)
    strategy: RedactionStrategy = RedactionStrategy.MASK
    field_names: list[str] = Field(default_factory=list)
    custom_patterns: list[str] = Field(
        default_factory=list, description="Custom regex patterns"
    )
    preserve_length: bool = False


# Regex patterns for common PII
PII_PATTERNS = {
    PIIPattern.SSN: r"\b\d{3}-\d{2}-\d{4}\b",
    PIIPattern.CREDIT_CARD: r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    PIIPattern.EMAIL: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    PIIPattern.PHONE: r"\b\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b",
    PIIPattern.IP_ADDRESS: r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    PIIPattern.API_KEY: r"\b[A-Za-z0-9_-]{32,}\b",  # Common API key length
}


class Redactor:
    """Redacts PII and sensitive data from checkpoint state.

    Example:
        >>> policy = RedactionPolicy(
        ...     patterns=[PIIPattern.EMAIL],
        ...     strategy=RedactionStrategy.MASK,
        ... )
        >>> redactor = Redactor(policy)
        >>> data = {"user": {"email": "user@example.com"}}
        >>> redacted = redactor.redact_state(data)
        >>> assert redacted["user"]["email"] == "****@*******.***"

    """

    def __init__(self, policy: RedactionPolicy):
        """Initialize redactor with policy.

        Args:
            policy: Redaction policy configuration.

        """
        self.policy = policy
        self._compiled_patterns: dict[PIIPattern, re.Pattern] = {}

        # Compile patterns
        for pattern in policy.patterns:
            if pattern in PII_PATTERNS:
                self._compiled_patterns[pattern] = re.compile(PII_PATTERNS[pattern])

        # Compile custom patterns
        self._custom_patterns: list[re.Pattern] = []
        for custom in policy.custom_patterns:
            self._custom_patterns.append(re.compile(custom))

    def redact_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Redact PII from state dictionary.

        Args:
            state: State dictionary to redact.

        Returns:
            Redacted state dictionary (deep copy).

        """
        if not self.policy.enabled:
            return state

        import copy

        redacted = copy.deepcopy(state)
        self._redact_recursive(redacted)
        return redacted

    def _redact_recursive(self, obj: Any, parent_key: str = "") -> None:
        """Recursively redact object in-place."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Check if field name should be redacted
                if key in self.policy.field_names:
                    obj[key] = self._apply_redaction(str(value))
                elif isinstance(value, str):
                    # Apply pattern matching to string values
                    obj[key] = self._redact_string_with_patterns(value)
                else:
                    self._redact_recursive(value, parent_key=key)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    obj[i] = self._redact_string_with_patterns(item)
                else:
                    self._redact_recursive(item, parent_key=parent_key)

    def _redact_string_with_patterns(self, text: str) -> str:
        """Apply pattern-based redaction to string.

        Args:
            text: Text to redact.

        Returns:
            Redacted text.

        """
        redacted = text

        # Apply pattern-based redaction
        for compiled_pattern in self._compiled_patterns.values():
            matches = compiled_pattern.findall(redacted)
            for match in matches:
                replacement = self._apply_redaction(match)
                redacted = redacted.replace(match, replacement)

        # Apply custom patterns
        for pattern in self._custom_patterns:
            matches = pattern.findall(redacted)
            for match in matches:
                replacement = self._apply_redaction(match)
                redacted = redacted.replace(match, replacement)

        return redacted

    def redact_string(self, text: str) -> str:
        """Redact PII patterns from string.

        Args:
            text: Text to redact.

        Returns:
            Redacted text.

        """
        if not self.policy.enabled:
            return text

        redacted = text

        # Apply pattern-based redaction
        for compiled_pattern in self._compiled_patterns.values():
            matches = compiled_pattern.findall(redacted)
            for match in matches:
                replacement = self._apply_redaction(match)
                redacted = redacted.replace(match, replacement)

        # Apply custom patterns
        for pattern in self._custom_patterns:
            matches = pattern.findall(redacted)
            for match in matches:
                replacement = self._apply_redaction(match)
                redacted = redacted.replace(match, replacement)

        return redacted

    def _apply_redaction(self, value: str) -> str:  # noqa: PLR0911
        """Apply redaction strategy to value.

        Args:
            value: Original value.

        Returns:
            Redacted value.

        """
        if self.policy.strategy == RedactionStrategy.REMOVE:
            return ""

        if self.policy.strategy == RedactionStrategy.HASH:
            import hashlib

            return hashlib.sha256(value.encode()).hexdigest()[:16]

        if self.policy.strategy == RedactionStrategy.PARTIAL:
            # Show first 3 and last 4 chars for SSN-like patterns
            if len(value) > PARTIAL_REDACTION_MIN_LENGTH:
                return f"{value[:3]}***{value[-4:]}"
            return "***"

        if self.policy.strategy == RedactionStrategy.MASK:
            if self.policy.preserve_length:
                # Replace alphanumeric with *, keep special chars
                return "".join("*" if c.isalnum() else c for c in value)
            return "***REDACTED***"

        return value


def redact_checkpoint_state(
    state: dict[str, Any], policy: RedactionPolicy
) -> dict[str, Any]:
    """Redact checkpoint state using policy.

    Args:
        state: State dictionary to redact.
        policy: Redaction policy.

    Returns:
        Redacted state dictionary.

    Example:
        >>> policy = RedactionPolicy(patterns=[PIIPattern.EMAIL])
        >>> state = {"user_email": "test@example.com"}
        >>> redacted = redact_checkpoint_state(state, policy)

    """
    redactor = Redactor(policy)
    return redactor.redact_state(state)
