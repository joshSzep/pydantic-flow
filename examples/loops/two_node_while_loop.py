"""Two-node while-loop example with plan and execute nodes."""

import asyncio

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import Route
from pydantic_flow import RunConfig
from pydantic_flow.core.routing import T_Route
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes import NodeWithInput


class WorkState(BaseModel):
    """State containing work progress."""

    iterations: int
    total: int


class PlanOutput(BaseModel):
    """Output from plan node."""

    plan: WorkState


class FullOutput(BaseModel):
    """Full output with both plan and execute results."""

    plan: WorkState
    execute: WorkState


class PlanNode(BaseNode[WorkState, WorkState]):
    """Planning node that prepares work."""

    async def run(self, input_data: WorkState) -> WorkState:
        """Plan the next iteration."""
        new_state = WorkState(
            iterations=input_data.iterations + 1, total=input_data.total
        )
        print(f"Planning iteration {new_state.iterations}")
        return new_state


class ExecuteNode(NodeWithInput[WorkState, WorkState]):
    """Execution node that performs work."""

    async def run(self, input_data: WorkState) -> WorkState:
        """Execute work and accumulate results."""
        new_total = input_data.total + (input_data.iterations * 10)
        new_state = WorkState(iterations=input_data.iterations, total=new_total)
        print(f"Executing: iteration={new_state.iterations}, total={new_total}")
        return new_state


def create_loop_router(max_iterations: int):
    """Create a router that loops until max iterations reached."""

    def router(state: BaseModel) -> T_Route:
        """Route back to plan or terminate with END."""
        execute_state = getattr(state, "execute", None)
        if execute_state and execute_state.iterations >= max_iterations:
            print(
                f"Reached {execute_state.iterations} iterations, "
                f"total={execute_state.total}"
            )
            print("Terminating with Route.END")
            return Route.END
        return "plan"

    return router


async def main() -> None:
    """Run the two-node while-loop example."""
    print("Two-Node While-Loop Example")
    print("=" * 50)

    flow = Flow(input_type=WorkState, output_type=FullOutput)

    plan_node = PlanNode(name="plan")
    execute_node = ExecuteNode(input=plan_node.output, name="execute")

    flow.add_nodes(plan_node, execute_node)
    flow.set_entry_nodes("plan")

    # Static edge ensures execute runs after plan in each iteration
    flow.add_edge("plan", "execute")

    router = create_loop_router(max_iterations=5)
    flow.add_conditional_edges("execute", router)

    compiled = flow.compile()

    config = RunConfig(max_steps=50, trace_iterations=True)
    result = await compiled.invoke(WorkState(iterations=0, total=0), config)

    print("\nFinal result:")
    print(f"  Iterations: {result.execute.iterations}")
    print(f"  Total: {result.execute.total}")


if __name__ == "__main__":
    asyncio.run(main())
