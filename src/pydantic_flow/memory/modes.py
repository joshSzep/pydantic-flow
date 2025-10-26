"""Memory modes for FlowNode sub-flow execution.

This module defines how conversation memory is handled when executing
sub-flows within FlowNodes.
"""

from enum import Enum


class MemoryMode(str, Enum):
    """Memory handling mode for FlowNode execution.

    Defines how conversation memory is propagated to sub-flows:

    - SHARED: Sub-flow uses parent's memory directly. Changes made by
      the sub-flow are visible to parent and affect the shared memory.
      Use for tightly coupled flows that should maintain conversation context.

    - ISOLATED: Sub-flow gets a new, separate memory instance.
      Optionally seeded with parent memory for context, but changes
      do not affect parent memory. Use for independent sub-tasks that
      need their own conversation context.

    - READONLY: Sub-flow gets read-only access to parent memory.
      Can read conversation history but cannot modify it. Useful for
      sub-flows that need context but shouldn't add to conversation history.
    """

    SHARED = "shared"
    ISOLATED = "isolated"
    READONLY = "readonly"
