"""Self-loop counter example demonstrating loop termination with Route.END."""

import asyncio

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import Route
from pydantic_flow import RunConfig
from pydantic_flow.core.routing import T_Route
from pydantic_flow.nodes import BaseNode


class CounterState(BaseModel):
    """State containing a counter."""

    n: int


class OutputState(BaseModel):
    """Output state with final tick value."""

    tick: CounterState


class TickNode(BaseNode[CounterState, CounterState]):
    """Node that increments the counter."""

    async def run(self, input_data: CounterState) -> CounterState:
        """Increment the counter by 1."""
        new_n = input_data.n + 1
        print(f"Tick: n = {new_n}")
        return CounterState(n=new_n)


def create_counter_router(max_count: int):
    """Create a router that loops until reaching max_count."""

    def router(state: BaseModel) -> T_Route:
        """Route to END when n >= max_count, otherwise loop back to tick."""
        tick_state = getattr(state, "tick", None)
        if tick_state and tick_state.n >= max_count:
            print(f"Counter reached {tick_state.n}, terminating with Route.END")
            return Route.END
        return "tick"

    return router


async def main() -> None:
    """Run the self-loop counter example."""
    print("Self-Loop Counter Example")
    print("=" * 50)

    flow = Flow(input_type=CounterState, output_type=OutputState)
    tick_node = TickNode(name="tick")
    flow.add_nodes(tick_node)
    flow.set_entry_nodes("tick")

    router = create_counter_router(max_count=5)
    flow.add_conditional_edges("tick", router)

    compiled = flow.compile()

    config = RunConfig(max_steps=50, trace_iterations=True)
    result = await compiled.invoke(CounterState(n=0), config)

    print(f"\nFinal result: n = {result.tick.n}")


if __name__ == "__main__":
    asyncio.run(main())
