"""Observer utilities for pydantic-ai agent streaming integration.

This module provides minimal observers and adapters that translate
pydantic-ai streaming events into our progress item vocabulary.
"""

from collections.abc import AsyncIterator
from typing import Any
import uuid

from pydantic_ai import Agent

from pydantic_flow.streaming.events import NonFatalError
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import TokenChunk


async def observe_agent_stream(
    agent: Agent[Any, Any],
    prompt: str,
    message_history: list[Any] | None = None,
    *,
    run_id: str | None = None,
    node_id: str = "",
) -> AsyncIterator[ProgressItem]:
    """Observe a pydantic-ai agent's streaming execution.

    This adapter translates the agent's streaming text output
    into our progress item vocabulary.

    Args:
        agent: The pydantic-ai Agent instance.
        prompt: The prompt to send to the agent.
        message_history: Optional message history to continue conversation.
        run_id: Optional run identifier.
        node_id: Identifier of the node making this call.

    Yields:
        Progress items representing the agent's streaming execution.

    """
    actual_run_id = run_id or str(uuid.uuid4())

    # Emit start
    yield StreamStart(
        run_id=actual_run_id,
        node_id=node_id,
        input_preview={"prompt": prompt[:100]},
    )

    try:
        # Run the agent and stream the result
        # pydantic-ai's streaming is available via result streaming
        async with agent.run_stream(prompt, message_history=message_history) as stream:
            token_index = 0
            async for chunk in stream.stream_text():
                yield TokenChunk(
                    text=chunk,
                    token_index=token_index,
                    run_id=actual_run_id,
                    node_id=node_id,
                )
                token_index += 1

            # Get the final result
            result = await stream.get_output()

        # Emit successful end with result preview
        result_preview = None
        if hasattr(result, "model_dump"):
            result_preview = result.model_dump()
        elif result is not None:
            result_preview = {"value": str(result)}

        yield StreamEnd(
            run_id=actual_run_id,
            node_id=node_id,
            result_preview=result_preview,
        )

    except Exception as e:
        # Emit error and re-raise
        yield NonFatalError(
            message=f"Agent stream failed: {e}",
            recoverable=False,
            run_id=actual_run_id,
            node_id=node_id,
        )
        raise


async def stream_agent_text(
    agent: Agent[Any, str],
    prompt: str,
    message_history: list[Any] | None = None,
) -> AsyncIterator[str]:
    """Stream text directly from a pydantic-ai agent.

    Simple helper for text-only streaming without progress events.

    Args:
        agent: The pydantic-ai Agent instance.
        prompt: The prompt to send to the agent.
        message_history: Optional message history to continue conversation.

    Yields:
        Text chunks from the agent.

    """
    async with agent.run_stream(prompt, message_history=message_history) as stream:
        async for chunk in stream.stream_text():
            yield chunk
