"""Two-node while-loop example with plan and execute nodes."""

import asyncio
from collections.abc import AsyncIterator

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import Route
from pydantic_flow import RunConfig
from pydantic_flow.nodes import BaseNode
from pydantic_flow.streaming import ProgressItem
from pydantic_flow.streaming import StreamEnd
from pydantic_flow.streaming import StreamStart


class WorkState(BaseModel):
    """State containing work progress."""

    iterations: int
    total: int


class PlanNode(BaseNode[WorkState, WorkState]):
    """Planning node that prepares work."""

    async def astream(self, input_data: WorkState) -> AsyncIterator[ProgressItem]:
        """Plan the next iteration."""
        yield StreamStart()
        new_state = WorkState(
            iterations=input_data.iterations + 1, total=input_data.total
        )
        print(f"Planning iteration {new_state.iterations}")
        yield StreamEnd(result=new_state)

    async def run(self, input_data: WorkState) -> WorkState:
        """Plan the next iteration."""
        new_state = WorkState(
            iterations=input_data.iterations + 1, total=input_data.total
        )
        print(f"Planning iteration {new_state.iterations}")
        return new_state


class ExecuteNode(BaseNode[WorkState, WorkState]):
    """Execution node that performs work."""

    async def astream(self, input_data: WorkState) -> AsyncIterator[ProgressItem]:
        """Execute work and accumulate results."""
        yield StreamStart()
        new_total = input_data.total + (input_data.iterations * 10)
        new_state = WorkState(iterations=input_data.iterations, total=new_total)
        print(f"Executing: iteration={new_state.iterations}, total={new_total}")
        yield StreamEnd(result=new_state)

    async def run(self, input_data: WorkState) -> WorkState:
        """Execute work and accumulate results."""
        new_total = input_data.total + (input_data.iterations * 10)
        new_state = WorkState(iterations=input_data.iterations, total=new_total)
        print(f"Executing: iteration={new_state.iterations}, total={new_total}")
        return new_state


def create_loop_router(max_iterations: int, plan_node: BaseNode):
    """Create a router that loops until max iterations reached."""

    def router(state: WorkState) -> BaseNode | Route:
        """Route back to plan or terminate with END."""
        if state.iterations >= max_iterations:
            print(f"Reached {state.iterations} iterations, total={state.total}")
            print("Terminating with Route.END")
            return Route.END
        print(f"Looping back to plan after iteration {state.iterations}")
        return plan_node

    return router


async def main() -> None:
    """Run the two-node while-loop example."""
    print("Two-Node While-Loop Example")
    print("=" * 50)

    plan_node = PlanNode()
    execute_node = ExecuteNode(inputs=(plan_node.output,))

    flow = Flow[WorkState, WorkState](input_type=WorkState, output_type=WorkState)
    flow.add_nodes(plan_node, execute_node)

    # Static edge ensures execute runs after plan in each iteration
    flow.add_edge(plan_node, execute_node)

    router = create_loop_router(max_iterations=5, plan_node=plan_node)
    flow.add_conditional_edges(execute_node, router)

    config = RunConfig(max_steps=50)
    result = None
    async for item in flow.astream(WorkState(iterations=0, total=0), config):
        if isinstance(item, StreamEnd):
            result = item.result

    print("\nFinal result:")
    if result and isinstance(result, WorkState):
        print(f"  Iterations: {result.iterations}")
        print(f"  Total: {result.total}")


if __name__ == "__main__":
    asyncio.run(main())
