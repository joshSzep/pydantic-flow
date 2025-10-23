"""Type Safety Demonstration for pydantic-flow.

This script demonstrates the comprehensive type safety features built into
the pydantic-flow framework.
"""

import asyncio

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import ToolNode


class WeatherQuery(BaseModel):
    """Input model for weather queries."""

    location: str


class WeatherInfo(BaseModel):
    """Output model for weather information."""

    location: str
    temperature: float
    description: str


class PersonalInfo(BaseModel):
    """Input model for personal information - different from WeatherQuery."""

    name: str
    age: int


class WeatherFlowResults(BaseModel):
    """Strongly typed results for weather flow."""

    weather_api: WeatherInfo


async def get_weather_data(query: WeatherQuery) -> WeatherInfo:
    """Mock weather API function."""
    return WeatherInfo(
        location=query.location,
        temperature=22.5,
        description="Sunny",
    )


def get_weather_data_sync(query: WeatherQuery) -> WeatherInfo:
    """Mock weather API function (synchronous version for ToolNode)."""
    return WeatherInfo(
        location=query.location,
        temperature=22.5,
        description="Sunny",
    )


async def demonstrate_typed_flow():
    """Demonstrate the improved type safety of Flow[InputT, OutputT]."""
    # Create a strongly typed flow that expects WeatherQuery inputs
    # and returns WeatherFlowResults
    weather_flow: Flow[WeatherQuery, WeatherFlowResults] = Flow(
        input_type=WeatherQuery, output_type=WeatherFlowResults
    )

    # Add a tool node that processes weather data
    weather_node = ToolNode[WeatherQuery, WeatherInfo](
        tool_func=get_weather_data_sync,
        name="weather_api",
    )

    weather_flow.add_nodes(weather_node)

    # This works - correct input type
    valid_input = WeatherQuery(location="Paris")
    results = await weather_flow.run(valid_input)
    print(f"Valid results: {results}")

    # Now results is strongly typed as WeatherFlowResults (BaseModel)
    # IDE will provide auto-completion for results.weather_api
    weather_info = results.weather_api
    print(f"Temperature in {weather_info.location}: {weather_info.temperature}°C")


async def demonstrate_basemodel_output_flow():
    """Show how flows work with BaseModel output types."""
    # Create a flow with explicit BaseModel output type
    basemodel_flow: Flow[WeatherQuery, WeatherFlowResults] = Flow(
        input_type=WeatherQuery, output_type=WeatherFlowResults
    )

    weather_node = ToolNode[WeatherQuery, WeatherInfo](
        tool_func=get_weather_data_sync,
        name="weather_api",
    )

    basemodel_flow.add_nodes(weather_node)

    # This works with explicit BaseModel output type
    query = WeatherQuery(location="London")
    results = await basemodel_flow.run(query)
    print(f"BaseModel results: {results}")

    # Access via attribute with full type safety
    weather_api_result = results.weather_api
    print(f"Weather info: {weather_api_result}")
    print(f"Temperature: {weather_api_result.temperature}°C")


def demonstrate_type_annotations():
    """Show different ways to create typed flows."""
    # Explicit type annotation with both input and output types
    typed_flow: Flow[WeatherQuery, WeatherFlowResults] = Flow(
        input_type=WeatherQuery, output_type=WeatherFlowResults
    )

    # Alternative BaseModel output type for different flow structure
    class SimpleWeatherResults(BaseModel):
        """Simple weather results model."""

        weather_api: WeatherInfo

    simple_flow: Flow[WeatherQuery, SimpleWeatherResults] = Flow(
        input_type=WeatherQuery, output_type=SimpleWeatherResults
    )

    # Type inference works when you immediately assign input-compatible nodes
    weather_node = ToolNode[WeatherQuery, WeatherInfo](
        tool_func=get_weather_data_sync,
        name="weather_api",
    )

    # The flow's add_nodes method now accepts BaseNode[InputT | Any, Any]
    # which means nodes can either take the flow's input type directly
    # or take output from other nodes (Any)
    typed_flow.add_nodes(weather_node)
    simple_flow.add_nodes(weather_node)

    print(f"Typed flow with {len(typed_flow.nodes)} nodes")
    print(f"Simple flow with {len(simple_flow.nodes)} nodes")


if __name__ == "__main__":
    print("=== pydantic-flow Type Safety Demonstration ===\n")

    print("1. Demonstrating strongly typed Flow[WeatherQuery, WeatherFlowResults]")
    asyncio.run(demonstrate_typed_flow())
    print()

    print("2. Demonstrating BaseModel output flows")
    asyncio.run(demonstrate_basemodel_output_flow())
    print()

    print("3. Demonstrating type annotation patterns")
    demonstrate_type_annotations()
    print()

    print("✅ Type safety improvements working correctly!")
    print("\nKey improvements:")
    print("- Flow[InputT, OutputT] constrains both input and output types")
    print("- InputT must be a BaseModel subclass")
    print("- OutputT must be a BaseModel subclass (strongly typed)")
    print("- Better IDE support with auto-completion for both inputs and outputs")
    print("- add_nodes() accepts BaseNode[InputT | Any, Any] for flexibility")
    print("- Dynamic BaseModel creation when output_type not specified")
    print("- Strong typing helps catch errors at compile time")
