"""Node mixins for composable functionality.

This module provides mixins that can be added to nodes to enable
additional capabilities like caching.
"""

from __future__ import annotations

from pydantic_flow.cache.base import CachePolicy


class CacheableNode:
    """Mixin for nodes that support caching.

    Add this mixin to node classes that want to opt into caching.
    The cache_policy attribute controls caching behavior.

    Example:
        ```python
        class MyNode(CacheableNode, BaseNode[InputT, OutputT]):
            def __init__(self, ..., cache_policy: CachePolicy | None = None):
                super().__init__(...)
                self.cache_policy = cache_policy
        ```

    """

    cache_policy: CachePolicy | None = None

    def is_cacheable(self) -> bool:
        """Check if this node has caching enabled.

        Returns:
            True if cache_policy is set and enabled.

        """
        return self.cache_policy is not None and self.cache_policy.enabled
