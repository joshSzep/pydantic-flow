"""Example demonstrating the pflow framework capabilities.

This example shows how to build a type-safe AI workflow using pflow,
including prompt nodes, parser nodes, tool nodes, and flow orchestration.
"""

import asyncio

from pydantic import BaseModel

from pflow import Flow
from pflow import ParserNode
from pflow import PromptNode
from pflow import ToolNode

# Temperature thresholds for weather recommendations
WARM_THRESHOLD = 25
PLEASANT_THRESHOLD = 15


# Define input and output models using Pydantic
class WeatherQuery(BaseModel):
    """Input model for weather queries."""

    location: str
    temperature_unit: str = "celsius"


class WeatherInfo(BaseModel):
    """Structured weather information."""

    temperature: float
    condition: str
    location: str
    humidity: int = 65


class WeatherSummary(BaseModel):
    """Weather summary with recommendations."""

    summary: str
    recommendation: str
    feels_like: str


class ApiFlowResults(BaseModel):
    """Results from the API-based weather flow."""

    weather_api: WeatherInfo
    summary_generator: WeatherSummary


class LlmFlowResults(BaseModel):
    """Results from the LLM-based weather flow."""

    weather_llm: str
    weather_parser: WeatherInfo
    summary_generator_2: WeatherSummary


# Define transformation functions
def parse_weather_response(weather_text: str) -> WeatherInfo:
    """Parse LLM weather response into structured data.

    In a real implementation, this would parse natural language.
    For demo purposes, we expect a simple format.
    """
    # Simple parser expecting "temp|condition|location" format
    try:
        parts = weather_text.split("|")
        return WeatherInfo(
            temperature=float(parts[0]),
            condition=parts[1].strip(),
            location=parts[2].strip(),
            humidity=65,  # default value
        )
    except ValueError, IndexError:
        # Fallback for unstructured responses
        return WeatherInfo(
            temperature=20.0,
            condition="Unknown",
            location="Unknown",
        )


def call_weather_api(query: WeatherQuery) -> WeatherInfo:
    """Mock weather API call.

    In a real implementation, this would call an actual weather service.
    """
    # Simulate different weather for different locations
    weather_data = {
        "paris": WeatherInfo(
            temperature=22.5,
            condition="sunny",
            location="Paris",
            humidity=45,
        ),
        "london": WeatherInfo(
            temperature=15.3,
            condition="cloudy",
            location="London",
            humidity=75,
        ),
        "tokyo": WeatherInfo(
            temperature=28.1,
            condition="partly cloudy",
            location="Tokyo",
            humidity=60,
        ),
    }

    location_key = query.location.lower()
    return weather_data.get(
        location_key,
        WeatherInfo(
            temperature=20.0,
            condition="unknown",
            location=query.location,
        ),
    )


def generate_summary(weather: WeatherInfo) -> WeatherSummary:
    """Generate a weather summary with recommendations."""
    temp = weather.temperature
    condition = weather.condition

    if temp > WARM_THRESHOLD:
        recommendation = "Perfect weather for outdoor activities!"
        feels_like = "warm"
    elif temp > PLEASANT_THRESHOLD:
        recommendation = "Great weather for a walk in the park."
        feels_like = "pleasant"
    else:
        recommendation = "Consider bringing a jacket."
        feels_like = "cool"

    summary = f"It's {temp}°C and {condition} in {weather.location}"

    return WeatherSummary(
        summary=summary,
        recommendation=recommendation,
        feels_like=feels_like,
    )


async def main():
    """Demonstrate the pflow framework with a weather workflow."""
    print("🌟 pflow Framework Demo - Weather Workflow")
    print("=" * 50)

    # Create workflow nodes
    print("\n1. Creating workflow nodes...")

    # Option 1: Direct API call workflow
    api_node = ToolNode[WeatherQuery, WeatherInfo](
        tool_func=call_weather_api,
        name="weather_api",
    )

    summary_node = ParserNode[WeatherInfo, WeatherSummary](
        parser_func=generate_summary,
        input=api_node.output,
        name="summary_generator",
    )

    # Option 2: LLM-based workflow (for demonstration)
    llm_node = PromptNode[WeatherQuery, str](
        prompt="22.5|sunny|{location}",  # Mock LLM response format
        name="weather_llm",
    )

    parser_node = ParserNode[str, WeatherInfo](
        parser_func=parse_weather_response,
        input=llm_node.output,
        name="weather_parser",
    )

    summary_node_2 = ParserNode[WeatherInfo, WeatherSummary](
        parser_func=generate_summary,
        input=parser_node.output,
        name="summary_generator_2",
    )

    print("✅ Nodes created successfully!")

    # Create and configure flows
    print("\n2. Setting up workflows...")

    # Flow 1: Direct API approach
    api_flow: Flow[WeatherQuery, ApiFlowResults] = Flow(
        input_type=WeatherQuery, output_type=ApiFlowResults
    )
    api_flow.add_nodes(api_node, summary_node)

    # Flow 2: LLM + Parser approach
    llm_flow: Flow[WeatherQuery, LlmFlowResults] = Flow(
        input_type=WeatherQuery, output_type=LlmFlowResults
    )
    llm_flow.add_nodes(llm_node, parser_node, summary_node_2)

    print(f"✅ API Flow execution order: {api_flow.get_execution_order()}")
    print(f"✅ LLM Flow execution order: {llm_flow.get_execution_order()}")

    # Test different locations
    locations = ["Paris", "London", "Tokyo", "Unknown City"]

    for location in locations:
        print(f"\n3. Testing weather for: {location}")
        print("-" * 30)

        query = WeatherQuery(location=location)

        # Run API-based flow
        print("📡 Running API-based workflow...")
        api_results = await api_flow.run(query)
        api_summary = api_results.summary_generator

        print(f"   Summary: {api_summary.summary}")
        print(f"   Feels like: {api_summary.feels_like}")
        print(f"   Recommendation: {api_summary.recommendation}")

        # Run LLM-based flow
        print("🤖 Running LLM-based workflow...")
        llm_results = await llm_flow.run(query)
        llm_summary = llm_results.summary_generator_2

        print(f"   Summary: {llm_summary.summary}")
        print(f"   Feels like: {llm_summary.feels_like}")
        print(f"   Recommendation: {llm_summary.recommendation}")

    print("\n4. Demonstrating type safety...")
    print("-" * 30)

    # Show that outputs are properly typed
    query = WeatherQuery(location="Paris")
    results = await api_flow.run(query)

    weather_info = results.weather_api
    summary = results.summary_generator

    print(f"✅ Weather info type: {type(weather_info).__name__}")
    print(f"✅ Summary type: {type(summary).__name__}")
    print(f"✅ Temperature (typed): {weather_info.temperature}°C")
    print(f"✅ Condition (typed): {weather_info.condition}")

    print("\n🎉 Demo completed successfully!")
    print("\nKey features demonstrated:")
    print("• Type-safe node composition with generics")
    print("• Automatic dependency resolution")
    print("• Multiple workflow patterns (API + LLM)")
    print("• Pydantic model validation")
    print("• DAG execution ordering")


if __name__ == "__main__":
    asyncio.run(main())
