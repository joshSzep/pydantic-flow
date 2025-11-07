"""AgentNode for LLM operations with user-supplied pydantic-ai agents."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
import uuid

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.memory import _active_flow_memory
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.observers import observe_agent_stream


class AgentNode[InputModel: BaseModel, OutputT](BaseNode[InputModel, OutputT]):
    """A streaming-native node that uses a pydantic-ai Agent.

    This node integrates user-supplied pydantic-ai agents with our streaming
    infrastructure, yielding tokens and progress items while the agent runs.

    Supports single or multiple inputs via the inputs parameter:
    - Single input: inputs=node.output
    - Multiple inputs: inputs=(node1.output, node2.output, ...)
    - Entry node: inputs=None

    Supports caching via the cache_policy attribute when used with a Flow
    that has a cache backend configured.
    """

    def __init__(
        self,
        agent: Agent[Any, OutputT],
        prompt_template: str | None = None,
        *,
        inputs: tuple[NodeOutput, ...] | None = None,
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
                           For multiple inputs, use {0}, {1}, etc. or field names.
            inputs: Optional tuple of inputs from other nodes:
                   - None: Entry node with no dependencies
                   - (node.output,): Single input dependency
                   - (node1.output, node2.output, ...): Multiple inputs (fan-in)
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.
            use_conversation_memory: Whether to use conversation memory from
                                   the active flow context. Default True.
            cache_policy: Optional cache policy for this node.

        """
        super().__init__(inputs, name, run_id, cache_policy)
        self.agent = agent
        self.prompt_template = prompt_template or ""
        self.use_conversation_memory = use_conversation_memory

    async def astream(
        self, input_data: InputModel | tuple[Any, ...]
    ) -> AsyncIterator[ProgressItem]:
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

    def _create_checkpoint(
        self, input_data: InputModel | tuple[Any, ...], item: ProgressItem
    ) -> Any:
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

    def _format_prompt(self, input_data: InputModel | tuple[Any, ...]) -> str:
        """Format the prompt template with input data.

        Args:
            input_data: The input model instance or tuple of inputs.

        Returns:
            Formatted prompt string.

        """
        if not self.prompt_template:
            return self._format_without_template(input_data)

        if isinstance(input_data, tuple):
            return self._format_template_with_tuple(input_data)

        return self.prompt_template.format(**input_data.model_dump())

    def _format_without_template(self, input_data: InputModel | tuple[Any, ...]) -> str:
        """Format input data without a template."""
        if isinstance(input_data, tuple):
            parts = []
            for item in input_data:
                if hasattr(item, "model_dump_json"):
                    parts.append(item.model_dump_json())
                else:
                    parts.append(str(item))
            return "\n\n".join(parts)

        if hasattr(input_data, "model_dump_json"):
            return input_data.model_dump_json()
        return str(input_data)

    def _format_template_with_tuple(self, input_data: tuple[Any, ...]) -> str:
        """Format template with tuple inputs, trying multiple strategies."""
        try:
            return self.prompt_template.format(*input_data)  # type: ignore
        except IndexError, KeyError:
            pass

        try:
            indexed = {str(i): val for i, val in enumerate(input_data)}
            return self.prompt_template.format(**indexed)  # type: ignore
        except IndexError, KeyError:
            pass

        try:
            combined = {}
            for _i, item in enumerate(input_data):
                if hasattr(item, "model_dump"):
                    combined.update(item.model_dump())
            return self.prompt_template.format(**combined)  # type: ignore
        except Exception:
            pass

        return "\n\n".join(str(val) for val in input_data)

    @classmethod
    def from_prompt(
        cls,
        model: str,
        prompt_template: str,
        *,
        system_prompt: str | None = None,
        output_type: type[OutputT] | None = None,
        inputs: tuple[NodeOutput, ...] | None = None,
        name: str | None = None,
        run_id: str | None = None,
        use_conversation_memory: bool = True,
        cache_policy: CachePolicy | None = None,
    ) -> AgentNode[InputModel, OutputT]:
        """Create an AgentNode with an internally-created agent.

        This factory method provides a convenient way to create an AgentNode
        without manually creating a pydantic-ai Agent first. Use this for
        simple cases. For more control, create the Agent yourself and pass
        it to AgentNode.__init__().

        Args:
            model: Model identifier (e.g., "openai:gpt-4").
            prompt_template: Prompt template string with {field} placeholders.
                           For multiple inputs, use {0}, {1}, etc.
            system_prompt: Optional system prompt/instructions.
            output_type: Expected output type - if BaseModel subclass, enables
                       structured output.
            inputs: Optional tuple of inputs from other nodes. Same as __init__.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.
            use_conversation_memory: Whether to use conversation memory. Default True.
            cache_policy: Optional cache policy for this node.

        Returns:
            AgentNode instance with internally-created agent.

        """
        instructions = system_prompt or "Be helpful and concise."

        # Create agent with or without structured output
        agent: Agent[Any, OutputT]
        if output_type is not None and isinstance(output_type, type):
            try:
                if issubclass(output_type, BaseModel):
                    agent = Agent(  # type: ignore[assignment]
                        model, instructions=instructions, output_type=output_type
                    )
                else:
                    agent = Agent(model, instructions=instructions)  # type: ignore[assignment]
            except TypeError:
                agent = Agent(model, instructions=instructions)  # type: ignore[assignment]
        else:
            agent = Agent(model, instructions=instructions)  # type: ignore[assignment]

        return cls(
            agent=agent,
            prompt_template=prompt_template,
            inputs=inputs,
            name=name,
            run_id=run_id,
            use_conversation_memory=use_conversation_memory,
            cache_policy=cache_policy,
        )
