"""Checkpoint compression utilities for reducing storage size.

This module provides compression and decompression utilities for checkpoint
node_states to reduce storage requirements while maintaining full fidelity.
"""

from __future__ import annotations

import gzip
import json
from typing import Any


def compress_node_states(node_states: dict[str, Any]) -> bytes:
    """Compress node states dictionary using gzip.

    Args:
        node_states: Dictionary of node execution results to compress.

    Returns:
        Compressed bytes representation of node_states.

    """
    json_str = json.dumps(node_states, default=str)
    json_bytes = json_str.encode("utf-8")
    return gzip.compress(json_bytes, compresslevel=6)


def decompress_node_states(compressed_data: bytes) -> dict[str, Any]:
    """Decompress node states from gzip-compressed bytes.

    Args:
        compressed_data: Gzip-compressed bytes from compress_node_states().

    Returns:
        Decompressed node_states dictionary.

    """
    json_bytes = gzip.decompress(compressed_data)
    json_str = json_bytes.decode("utf-8")
    return json.loads(json_str)


def calculate_compression_ratio(original: dict[str, Any], compressed: bytes) -> float:
    """Calculate compression ratio for node states.

    Args:
        original: Original uncompressed node_states dictionary.
        compressed: Compressed bytes from compress_node_states().

    Returns:
        Compression ratio (original_size / compressed_size).
        Higher values indicate better compression.

    """
    original_size = len(json.dumps(original, default=str).encode("utf-8"))
    compressed_size = len(compressed)
    if compressed_size == 0:
        return 0.0
    return original_size / compressed_size
