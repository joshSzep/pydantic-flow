"""Tests for JSON model parser."""

import json

from pydantic import BaseModel
from pydantic import ValidationError
import pytest

from pydantic_flow.prompt.parsers import JsonModelParser

EXPECTED_TEMP = 72.5
EXPECTED_ITEM_COUNT = 2
EXPECTED_VALUE = 2


class WeatherInfo(BaseModel):
    """Weather information model."""

    temperature: float
    condition: str
    location: str


class TestJsonModelParser:
    """Test JSON model parser."""

    async def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON into model."""
        parser = JsonModelParser[WeatherInfo](WeatherInfo)
        result = await parser.parse(
            '{"temperature": 72.5, "condition": "sunny", "location": "NYC"}'
        )

        assert isinstance(result, WeatherInfo)
        assert result.temperature == EXPECTED_TEMP
        assert result.condition == "sunny"
        assert result.location == "NYC"

    async def test_parse_invalid_json(self) -> None:
        """Test error on invalid JSON."""
        parser = JsonModelParser[WeatherInfo](WeatherInfo)

        with pytest.raises(json.JSONDecodeError):
            await parser.parse("{invalid json}")

    async def test_parse_validation_error(self) -> None:
        """Test error when JSON doesn't match model schema."""
        parser = JsonModelParser[WeatherInfo](WeatherInfo)

        with pytest.raises(ValidationError):
            await parser.parse('{"temperature": "not a number"}')

    async def test_parse_missing_field(self) -> None:
        """Test error when required field is missing."""
        parser = JsonModelParser[WeatherInfo](WeatherInfo)

        with pytest.raises(ValidationError):
            await parser.parse('{"temperature": 72.5, "condition": "sunny"}')

    async def test_parse_extra_field_strict(self) -> None:
        """Test that extra fields are allowed in strict mode."""
        parser = JsonModelParser[WeatherInfo](WeatherInfo, strict=True)

        result = await parser.parse(
            '{"temperature": 72.5, "condition": "sunny", '
            '"location": "NYC", "extra": "ignored"}'
        )
        assert isinstance(result, WeatherInfo)

    async def test_parse_non_strict(self) -> None:
        """Test non-strict parsing."""
        parser = JsonModelParser[WeatherInfo](WeatherInfo, strict=False)

        result = await parser.parse(
            '{"temperature": 72.5, "condition": "sunny", "location": "NYC"}'
        )
        assert isinstance(result, WeatherInfo)

    async def test_parse_with_type_coercion(self) -> None:
        """Test that type coercion works."""
        parser = JsonModelParser[WeatherInfo](WeatherInfo, strict=False)

        result = await parser.parse(
            '{"temperature": "72.5", "condition": "sunny", "location": "NYC"}'
        )
        assert result.temperature == EXPECTED_TEMP

    async def test_parse_nested_model(self) -> None:
        """Test parsing nested model."""

        class Location(BaseModel):
            city: str
            state: str

        class DetailedWeather(BaseModel):
            temperature: float
            location: Location

        parser = JsonModelParser[DetailedWeather](DetailedWeather)
        result = await parser.parse(
            '{"temperature": 72.5, "location": {"city": "NYC", "state": "NY"}}'
        )

        assert result.temperature == EXPECTED_TEMP
        assert result.location.city == "NYC"
        assert result.location.state == "NY"

    async def test_parse_list_of_items(self) -> None:
        """Test parsing JSON array."""

        class Item(BaseModel):
            name: str
            value: int

        class ItemList(BaseModel):
            items: list[Item]

        parser = JsonModelParser[ItemList](ItemList)
        result = await parser.parse(
            '{"items": [{"name": "a", "value": 1}, {"name": "b", "value": 2}]}'
        )

        assert len(result.items) == EXPECTED_ITEM_COUNT
        assert result.items[0].name == "a"
        assert result.items[1].value == EXPECTED_VALUE
