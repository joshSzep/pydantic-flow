"""Streaming-native PromptNode for LLM operations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
import uuid

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.memory import _active_flow_memory
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.nodes.mixins import CacheableNode
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import GenericResult
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.core_events import TokenChunk
from pydantic_flow.streaming.observers import observe_agent_stream
from pydantic_flow.streaming.system_events import NonFatalError


class AgentNode[InputModel: BaseModel, OutputT](
    CacheableNode, NodeWithInput[InputModel, OutputT]
):
    """A streaming-native node that uses a pydantic-ai Agent.

    This node integrates user-supplied pydantic-ai agents with our streaming
    infrastructure, yielding tokens and progress items while the agent runs.

    Supports caching via the cache_policy attribute when used with a Flow
    that has a cache backend configured.
    """

    def __init__(
        self,
        agent: Agent[Any, OutputT],
        prompt_template: str | None = None,
        *,
        input: Any = None,
        name: str | None = None,
        run_id: str | None = None,
        use_conversation_memory: bool = True,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize an AgentNode.

        Args:
            agent: The pydantic-ai Agent instance to use.
            prompt_template: Optional prompt template string. Uses {field}
                           syntax for variable interpolation from input.
            input: Optional input from another node's output.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.
            use_conversation_memory: Whether to use conversation memory from
                                   the active flow context. Default True.
            cache_policy: Optional cache policy for this node.

        """
        super().__init__(input, name, run_id)
        self.agent = agent
        self.prompt_template = prompt_template or ""
        self.use_conversation_memory = use_conversation_memory
        self.cache_policy = cache_policy

    async def astream(self, input_data: InputModel) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the LLM call.

        Yields:
            StreamStart, TokenChunk items during generation, and StreamEnd
            with the final result.

        """
        # Format prompt from input data
        prompt = self._format_prompt(input_data)

        # Get conversation memory from context if enabled
        message_history = None
        if self.use_conversation_memory:
            memory = _active_flow_memory.get()
            if memory is not None:
                message_history = memory.get()

        # Use observer to translate agent stream to our progress items
        async for item in observe_agent_stream(
            self.agent,
            prompt,
            message_history=message_history,
            run_id=self.run_id or str(uuid.uuid4()),
            node_id=self.name,
        ):
            # Attach interrupt handlers and check
            decision = await self._check_interrupt_handlers(item)
            if decision.should_interrupt:
                raise InterruptionRequested(
                    snapshot=self._create_checkpoint(input_data, item),
                    decision=decision,
                )
            yield item

    def _create_checkpoint(self, input_data: InputModel, item: ProgressItem) -> Any:
        """Create a checkpoint for resumption (placeholder).

        This will be fully implemented when Flow-level checkpointing is added.

        Args:
            input_data: Current input data.
            item: Progress item at interruption point.

        Returns:
            Placeholder checkpoint data.

        """
        # For now, return minimal checkpoint info
        # Full implementation will happen in Phase 3
        return {
            "node_id": self.name,
            "run_id": self.run_id,
            "item_type": item.type,
        }

    def _format_prompt(self, input_data: InputModel) -> str:
        """Format the prompt template with input data.

        Args:
            input_data: The input model instance.

        Returns:
            Formatted prompt string.

        """
        if not self.prompt_template:
            # No template, use input directly
            if hasattr(input_data, "model_dump_json"):
                return input_data.model_dump_json()
            return str(input_data)

        # Format template with input fields
        return self.prompt_template.format(**input_data.model_dump())


class LLMNode[InputModel: BaseModel, OutputModel: BaseModel](
    CacheableNode, NodeWithInput[InputModel, OutputModel]
):
    """A streaming-native LLM node with structured output.

    This node wraps a pydantic-ai agent and provides streaming of both
    tokens and partial structured fields during generation.
    """

    def __init__(
        self,
        agent: Agent[Any, OutputModel],
        prompt_template: str,
        *,
        input: Any = None,
        name: str | None = None,
        run_id: str | None = None,
        use_conversation_memory: bool = True,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize an LLMNode.

        Args:
            agent: The pydantic-ai Agent instance configured for structured output.
            prompt_template: Prompt template string with {field} placeholders.
            input: Optional input from another node's output.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.
            use_conversation_memory: Whether to use conversation memory from
                                   the active flow context. Default True.
            cache_policy: Optional cache policy for this node.

        """
        super().__init__(input, name, run_id)
        self.agent = agent
        self.prompt_template = prompt_template
        self.use_conversation_memory = use_conversation_memory
        self.cache_policy = cache_policy
        self.use_conversation_memory = use_conversation_memory

    async def astream(self, input_data: InputModel) -> AsyncIterator[ProgressItem]:
        """Stream progress items including tokens and partial fields.

        Yields:
            StreamStart, TokenChunk during generation, and StreamEnd with
            validated structured output.

        """
        prompt = self.prompt_template.format(**input_data.model_dump())
        actual_run_id = self.run_id or str(uuid.uuid4())

        # Get conversation memory from context if enabled
        message_history = None
        if self.use_conversation_memory:
            memory = _active_flow_memory.get()
            if memory is not None:
                message_history = memory.get()

        start_item = StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview={"prompt": prompt[:100]},
        )
        decision = await self._check_interrupt_handlers(start_item)
        if decision.should_interrupt:
            raise InterruptionRequested(
                snapshot=self._create_checkpoint(input_data, start_item),
                decision=decision,
            )
        yield start_item

        try:
            # Stream from agent with message history
            async with self.agent.run_stream(
                prompt, message_history=message_history
            ) as stream:
                token_index = 0
                async for chunk in stream.stream_text():
                    token_item = TokenChunk(
                        text=chunk,
                        token_index=token_index,
                        run_id=actual_run_id,
                        node_id=self.name,
                    )
                    decision = await self._check_interrupt_handlers(token_item)
                    if decision.should_interrupt:
                        raise InterruptionRequested(
                            snapshot=self._create_checkpoint(input_data, token_item),
                            decision=decision,
                        )
                    yield token_item
                    token_index += 1

                # Get final structured result
                result = await stream.get_output()

            # Emit end with result as BaseModel
            if isinstance(result, BaseModel):
                result_model = result
            else:
                result_model = GenericResult(value=result)

            end_item = StreamEnd(
                run_id=actual_run_id,
                node_id=self.name,
                result=result_model,
            )
            decision = await self._check_interrupt_handlers(end_item)
            if decision.should_interrupt:
                raise InterruptionRequested(
                    snapshot=self._create_checkpoint(input_data, end_item),
                    decision=decision,
                )
            yield end_item

        except InterruptionRequested:
            # Re-raise interruption requests
            raise
        except Exception as e:
            yield NonFatalError(
                message=f"LLM execution failed: {e}",
                recoverable=False,
                run_id=actual_run_id,
                node_id=self.name,
            )
            raise

    def _create_checkpoint(self, input_data: InputModel, item: ProgressItem) -> Any:
        """Create a checkpoint for resumption (placeholder).

        This will be fully implemented when Flow-level checkpointing is added.

        Args:
            input_data: Current input data.
            item: Progress item at interruption point.

        Returns:
            Placeholder checkpoint data.

        """
        # For now, return minimal checkpoint info
        # Full implementation will happen in Phase 3
        return {
            "node_id": self.name,
            "run_id": self.run_id,
            "item_type": item.type,
        }
