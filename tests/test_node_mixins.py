"""Tests for node mixins."""

from datetime import timedelta

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.nodes.mixins import CacheableNode


def test_cacheable_node_no_policy():
    """Test CacheableNode with no policy set."""

    class TestNode(CacheableNode):
        pass

    node = TestNode()
    assert node.cache_policy is None
    assert node.is_cacheable() is False


def test_cacheable_node_with_disabled_policy():
    """Test CacheableNode with disabled policy."""

    class TestNode(CacheableNode):
        def __init__(self, cache_policy: CachePolicy | None = None):
            self.cache_policy = cache_policy

    policy = CachePolicy(enabled=False, ttl=timedelta(seconds=60))
    node = TestNode(cache_policy=policy)
    assert node.cache_policy is not None
    assert node.is_cacheable() is False


def test_cacheable_node_with_enabled_policy():
    """Test CacheableNode with enabled policy."""

    class TestNode(CacheableNode):
        def __init__(self, cache_policy: CachePolicy | None = None):
            self.cache_policy = cache_policy

    policy = CachePolicy(enabled=True, ttl=timedelta(seconds=60))
    node = TestNode(cache_policy=policy)
    assert node.cache_policy is not None
    assert node.is_cacheable() is True
