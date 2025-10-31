"""Tests for checkpoint history and time-travel debugging features."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.memory import InMemoryCheckpointStore
from pydantic_flow.hitl.checkpoints.sqlite import SQLiteCheckpointStore
from pydantic_flow.hitl.checkpoints.sqlite import SQLiteCheckpointStoreConfig
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.streaming.base import ProgressItem


class Input(BaseModel):
    """Test input model."""

    value: int


class Output(BaseModel):
    """Test output model."""

    result: int


class AddTenNode(BaseNode[Input, Output]):
    """Node that adds 10 to the input value."""

    def __init__(self, **kwargs: Any):
        """Initialize the node."""
        super().__init__(name="add_ten", **kwargs)

    async def run(self, input_data: Input) -> Output:
        """Add 10 to value."""
        return Output(result=input_data.value + 10)

    async def astream(self, input_data: Input) -> AsyncIterator[ProgressItem]:
        """Stream execution."""
        from pydantic_flow.streaming.core_events import StreamEnd
        from pydantic_flow.streaming.core_events import StreamStart

        yield StreamStart(run_id="", node_id=self.name)
        result = await self.run(input_data)
        yield StreamEnd(
            run_id="", node_id=self.name, result_preview=result.model_dump()
        )


class MultiplyTwoNode(BaseNode[Output, Output]):
    """Node that multiplies the input by 2."""

    def __init__(self, **kwargs: Any):
        """Initialize the node."""
        super().__init__(name="multiply_two", **kwargs)
        from pydantic_flow.nodes.base import NodeOutput

        self.input: NodeOutput[Output] | None = None

    async def run(self, input_data: Output) -> Output:
        """Multiply by 2."""
        return Output(result=input_data.result * 2)

    async def astream(self, input_data: Output) -> AsyncIterator[ProgressItem]:
        """Stream execution."""
        from pydantic_flow.streaming.core_events import StreamEnd
        from pydantic_flow.streaming.core_events import StreamStart

        yield StreamStart(run_id="", node_id=self.name)
        result = await self.run(input_data)
        yield StreamEnd(
            run_id="", node_id=self.name, result_preview=result.model_dump()
        )


@pytest.fixture
def simple_flow() -> Flow[Input, Output]:
    """Create a simple two-node flow for testing."""
    flow = Flow(input_type=Input, output_type=Output)

    node1 = AddTenNode()
    node2 = MultiplyTwoNode()
    node2.input = node1.output

    flow.add_nodes(node1, node2)
    flow.add_edge(node1, node2)

    return flow


@pytest.mark.asyncio
class TestCheckpointStoreHistory:
    """Test checkpoint store history methods."""

    async def test_count_checkpoints_empty(self) -> None:
        """Test counting checkpoints when none exist."""
        store = InMemoryCheckpointStore()
        count = await store.count_checkpoints(RunId("nonexistent"))
        assert count == 0

    async def test_get_checkpoint_history_empty(self) -> None:
        """Test getting checkpoint history when none exist."""
        store = InMemoryCheckpointStore()
        history = await store.get_checkpoint_history(RunId("nonexistent"))
        assert len(history) == 0

    async def test_count_and_history_after_saves(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test count and history after saving multiple checkpoints."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="test-run",
        )

        # Run flow to create checkpoints
        result = await simple_flow.run(Input(value=5), config=config)
        assert result.result == 30  # (5 + 10) * 2

        # Count checkpoints
        count = await store.count_checkpoints(RunId("test-run"))
        assert count > 0

        # Get history
        history = await store.get_checkpoint_history(RunId("test-run"))
        assert len(history) > 0
        assert len(history) == count

    async def test_history_sorted_newest_first(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test that checkpoint history is sorted newest first."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="test-run",
        )

        await simple_flow.run(Input(value=5), config=config)

        history = await store.get_checkpoint_history(RunId("test-run"), limit=10)

        # Verify sorting
        for i in range(len(history) - 1):
            assert history[i].created_at >= history[i + 1].created_at

    async def test_history_limit(self, simple_flow: Flow[Input, Output]) -> None:
        """Test that history respects the limit parameter."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="test-run",
        )

        await simple_flow.run(Input(value=5), config=config)

        # Get all checkpoints
        all_history = await store.get_checkpoint_history(RunId("test-run"), limit=100)
        total_count = len(all_history)

        # Get limited checkpoints
        if total_count > 1:
            limited_history = await store.get_checkpoint_history(
                RunId("test-run"), limit=1
            )
            assert len(limited_history) == 1
            assert limited_history[0].id == all_history[0].id  # Newest first

    async def test_sqlite_store_count_and_history(
        self, simple_flow: Flow[Input, Output], tmp_path
    ) -> None:
        """Test count and history with SQLite store."""
        from pydantic_flow.core.durability import DurabilityMode

        db_path = tmp_path / "test_checkpoint_history.db"
        store = SQLiteCheckpointStore(SQLiteCheckpointStoreConfig(db_path=db_path))

        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="test-run",
        )

        await simple_flow.run(Input(value=5), config=config)

        # Count checkpoints
        count = await store.count_checkpoints(RunId("test-run"))
        assert count > 0

        # Get history
        history = await store.get_checkpoint_history(RunId("test-run"))
        assert len(history) == count

        # Verify SQL index usage (newest first)
        for i in range(len(history) - 1):
            assert history[i].created_at >= history[i + 1].created_at


@pytest.mark.asyncio
class TestFlowListCheckpoints:
    """Test Flow.list_checkpoints() method."""

    async def test_list_checkpoints_empty(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test listing checkpoints when none exist."""
        store = InMemoryCheckpointStore()
        checkpoints = await simple_flow.list_checkpoints("nonexistent", store)
        assert len(checkpoints) == 0

    async def test_list_checkpoints_after_run(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test listing checkpoints after a successful run."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="test-run",
        )

        await simple_flow.run(Input(value=5), config=config)

        checkpoints = await simple_flow.list_checkpoints("test-run", store)
        assert len(checkpoints) > 0

        # Verify all checkpoints are for the correct run
        for checkpoint in checkpoints:
            assert checkpoint.checkpoint.run_id == "test-run"

    async def test_list_checkpoints_limit(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test that list_checkpoints respects the limit parameter."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="test-run",
        )

        await simple_flow.run(Input(value=5), config=config)

        # Get all checkpoints
        all_checkpoints = await simple_flow.list_checkpoints(
            "test-run", store, limit=100
        )

        if len(all_checkpoints) > 1:
            # Get limited checkpoints
            limited = await simple_flow.list_checkpoints("test-run", store, limit=1)
            assert len(limited) == 1
            assert limited[0].id == all_checkpoints[0].id


@pytest.mark.asyncio
class TestFlowReplayFromCheckpoint:
    """Test Flow.replay_from_checkpoint() method."""

    async def test_replay_creates_new_run_id(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test that replay creates a new run_id."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="original-run",
        )

        # Original run
        result1 = await simple_flow.run(Input(value=5), config=config)
        assert result1.result == 30

        # Get checkpoint - use last one (oldest, should be intermediate)
        checkpoints = await simple_flow.list_checkpoints("original-run", store)
        assert len(checkpoints) > 0
        checkpoint = checkpoints[
            -1
        ].checkpoint  # Last checkpoint (oldest, intermediate)

        # Replay with SYNC mode so checkpoints are created
        replay_config = RunConfig(durability_mode=DurabilityMode.SYNC)
        result2 = await simple_flow.replay_from_checkpoint(
            checkpoint, store, config=replay_config
        )
        assert result2.result == 30

        # Verify new run_id was created
        all_checkpoints, _ = await store.list(CheckpointQuery(limit=100))
        run_ids = {env.checkpoint.run_id for env in all_checkpoints}
        assert len(run_ids) >= 2, f"Expected at least 2 run_ids, got: {run_ids}"
        assert "original-run" in run_ids

    async def test_replay_with_new_config(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test replay with different configuration."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config1 = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="run-1",
        )

        # Original run
        await simple_flow.run(Input(value=5), config=config1)

        # Get checkpoint
        checkpoints = await simple_flow.list_checkpoints("run-1", store)
        checkpoint = checkpoints[0].checkpoint

        # Replay with different durability mode
        config2 = RunConfig(durability_mode=DurabilityMode.ASYNC)
        result = await simple_flow.replay_from_checkpoint(
            checkpoint, store, config=config2
        )
        assert result.result == 30

    async def test_replay_new_input_not_implemented(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test that new_input parameter raises NotImplementedError."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="test-run",
        )

        await simple_flow.run(Input(value=5), config=config)
        checkpoints = await simple_flow.list_checkpoints("test-run", store)
        checkpoint = checkpoints[0].checkpoint

        # Attempt to replay with new input
        with pytest.raises(NotImplementedError):
            await simple_flow.replay_from_checkpoint(
                checkpoint, store, new_input=Input(value=10)
            )


@pytest.mark.asyncio
class TestFlowForkFromCheckpoint:
    """Test Flow.fork_from_checkpoint() method."""

    async def test_fork_creates_new_run_id(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test that fork creates a new run_id."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="original-run",
        )

        # Original run
        await simple_flow.run(Input(value=5), config=config)

        # Get checkpoint after first node
        checkpoints = await simple_flow.list_checkpoints("original-run", store)
        # Find checkpoint with add_ten completed
        checkpoint = None
        for env in checkpoints:
            if "add_ten" in env.checkpoint.execution_progress:
                if env.checkpoint.execution_progress["add_ten"] == "completed":
                    checkpoint = env.checkpoint
                    break

        if checkpoint is None:
            pytest.skip("Could not find suitable checkpoint for forking")
        assert checkpoint is not None

        # Fork with modified first node output
        modified = Output(result=100)
        result = await simple_flow.fork_from_checkpoint(
            checkpoint, store, modifications={"add_ten": modified}
        )

        # Result should be modified value * 2
        assert result.result == 200  # 100 * 2

    async def test_fork_modifies_node_states(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test that fork correctly modifies node states."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="test-run",
        )

        await simple_flow.run(Input(value=5), config=config)

        checkpoints = await simple_flow.list_checkpoints("test-run", store)
        checkpoint = None
        for env in checkpoints:
            if "add_ten" in env.checkpoint.node_states:
                checkpoint = env.checkpoint
                break

        if checkpoint is None:
            pytest.skip("Could not find suitable checkpoint for forking")
        assert checkpoint is not None

        # Fork and verify original checkpoint unchanged
        original_value = checkpoint.node_states["add_ten"].result
        modified = Output(result=999)

        await simple_flow.fork_from_checkpoint(
            checkpoint, store, modifications={"add_ten": modified}
        )

        # Original checkpoint should be unchanged
        assert checkpoint.node_states["add_ten"].result == original_value

    async def test_fork_invalid_node_raises_error(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test that forking with invalid node name raises KeyError."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="test-run",
        )

        await simple_flow.run(Input(value=5), config=config)
        checkpoints = await simple_flow.list_checkpoints("test-run", store)
        checkpoint = checkpoints[0].checkpoint

        # Attempt to fork with non-existent node
        with pytest.raises(KeyError, match="not found in checkpoint"):
            await simple_flow.fork_from_checkpoint(
                checkpoint, store, modifications={"nonexistent_node": Output(result=1)}
            )

    async def test_fork_multiple_modifications(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test forking with multiple node modifications."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="test-run",
        )

        await simple_flow.run(Input(value=5), config=config)

        checkpoints = await simple_flow.list_checkpoints("test-run", store)
        # Find a checkpoint with both nodes completed
        checkpoint = None
        for env in checkpoints:
            progress = env.checkpoint.execution_progress
            if (
                "add_ten" in progress
                and progress["add_ten"] == "completed"
                and "multiply_two" in progress
                and progress["multiply_two"] == "completed"
            ):
                checkpoint = env.checkpoint
                break

        if checkpoint is None:
            pytest.skip("Could not find checkpoint with both nodes completed")
        assert checkpoint is not None

        # Fork with both nodes modified (though this is unusual)
        modifications = {
            "add_ten": Output(result=50),
            "multiply_two": Output(result=200),
        }

        result = await simple_flow.fork_from_checkpoint(
            checkpoint, store, modifications=modifications
        )

        # Since both nodes are completed, result should be from multiply_two
        assert result.result == 200


@pytest.mark.asyncio
class TestCheckpointHistoryIntegration:
    """Integration tests for checkpoint history features."""

    async def test_full_workflow(self, simple_flow: Flow[Input, Output]) -> None:
        """Test complete workflow: run, list, replay, fork."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()
        config = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="original-run",
        )

        # 1. Original run
        result1 = await simple_flow.run(Input(value=5), config=config)
        assert result1.result == 30

        # 2. List checkpoints
        checkpoints = await simple_flow.list_checkpoints("original-run", store)
        assert len(checkpoints) > 0

        # 3. Replay from checkpoint
        checkpoint = checkpoints[0].checkpoint
        result2 = await simple_flow.replay_from_checkpoint(checkpoint, store)
        assert result2.result == 30

        # 4. Fork from checkpoint
        checkpoint_for_fork = None
        for env in checkpoints:
            if "add_ten" in env.checkpoint.node_states:
                checkpoint_for_fork = env.checkpoint
                break

        if checkpoint_for_fork:
            result3 = await simple_flow.fork_from_checkpoint(
                checkpoint_for_fork,
                store,
                modifications={"add_ten": Output(result=50)},
            )
            assert result3.result == 100  # 50 * 2

        # 5. Verify all runs are stored
        count = await store.count_checkpoints(RunId("original-run"))
        assert count > 0

    async def test_multiple_runs_isolated(
        self, simple_flow: Flow[Input, Output]
    ) -> None:
        """Test that multiple runs maintain isolated checkpoints."""
        from pydantic_flow.core.durability import DurabilityMode

        store = InMemoryCheckpointStore()

        # Run 1
        config1 = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="run-1",
        )
        await simple_flow.run(Input(value=5), config=config1)

        # Run 2
        config2 = RunConfig(
            durability_mode=DurabilityMode.SYNC,
            checkpoint_store=store,
            run_id="run-2",
        )
        await simple_flow.run(Input(value=10), config=config2)

        # Verify isolation
        checkpoints1 = await simple_flow.list_checkpoints("run-1", store)
        checkpoints2 = await simple_flow.list_checkpoints("run-2", store)

        assert len(checkpoints1) > 0
        assert len(checkpoints2) > 0

        # Verify no cross-contamination
        for env in checkpoints1:
            assert env.checkpoint.run_id == "run-1"
        for env in checkpoints2:
            assert env.checkpoint.run_id == "run-2"

        # Count should match
        count1 = await store.count_checkpoints(RunId("run-1"))
        count2 = await store.count_checkpoints(RunId("run-2"))
        assert count1 == len(checkpoints1)
        assert count2 == len(checkpoints2)
