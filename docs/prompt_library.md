# Prompt Library

Type-safe prompt management for `pydantic-flow`, supporting multiple template engines with structured output parsing.

## Features

- **Multiple Template Engines**: f-string, Jinja2, and Mustache
- **Type-Safe**: Full type annotations with Pydantic models
- **Structured Output**: Built-in parsers for JSON, delimited text, and custom formats
- **Chat-First**: Native support for role-based chat messages
- **Escape Policies**: HTML escaping and custom policies for security
- **Observability**: OpenTelemetry integration for tracing and monitoring

## Quick Start

```python
from pydantic import BaseModel
from pydantic_flow.prompt import (
    PromptTemplate,
    ChatPromptTemplate,
    ChatMessage,
    ChatRole,
    TemplateFormat,
    JoinStrategy,
    JsonModelParser,
)

# Define your input and output models
class WeatherQuery(BaseModel):
    location: str
    unit: str = "celsius"

class WeatherInfo(BaseModel):
    temperature: float
    condition: str
    location: str

# Simple prompt template
template = PromptTemplate[WeatherQuery, str](
    template="What's the weather in {location} ({unit})?",
    format=TemplateFormat.F_STRING,
    input_model=WeatherQuery,
)

result = template.render(WeatherQuery(location="Paris"))
# Output: "What's the weather in Paris (celsius)?"

# With structured output parsing
json_template = PromptTemplate[WeatherQuery, WeatherInfo](
    template='{{"temp": 72.5, "condition": "sunny", "location": "{location}"}}',
    format=TemplateFormat.F_STRING,
    input_model=WeatherQuery,
    output_parser=JsonModelParser(WeatherInfo),
)

parsed = await json_template.render_and_parse(
    WeatherQuery(location="NYC")
)
# Returns validated WeatherInfo instance
```

## Template Engines

### F-String (Default)

Python-style string formatting with format specifications:

```python
template = PromptTemplate[QueryInput, str](
    template="Temperature: {temp:.1f}°C, Count: {n:03d}",
    format=TemplateFormat.F_STRING,
    input_model=QueryInput,
)
```

### Jinja2

Full Jinja2 template support with sandboxed environment:

```python
template = PromptTemplate[QueryInput, str](
    template="""
    {% if admin %}
        Admin dashboard for {{ user }}
    {% else %}
        User dashboard for {{ user }}
    {% endif %}
    """,
    format=TemplateFormat.JINJA2,
    input_model=QueryInput,
)
```

### Mustache

Logic-less templates for simple substitutions:

```python
template = PromptTemplate[QueryInput, str](
    template="Hello {{name}}, welcome to {{place}}!",
    format=TemplateFormat.MUSTACHE,
    input_model=QueryInput,
)
```

## Chat Messages

Role-based chat templates with multiple join strategies:

```python
chat_template = ChatPromptTemplate[QueryInput, str](
    messages=[
        ChatMessage(
            role=ChatRole.SYSTEM,
            content="You are a helpful weather assistant."
        ),
        ChatMessage(
            role=ChatRole.USER,
            content="What's the weather in {location}?"
        ),
    ],
    format=TemplateFormat.F_STRING,
    input_model=QueryInput,
)

# Render messages
rendered = chat_template.render_messages(QueryInput(location="Tokyo"))

# Join with different strategies
openai_format = chat_template.join(JoinStrategy.OPENAI, rendered)
# Output:
# system: You are a helpful weather assistant.
# user: What's the weather in Tokyo?

anthropic_format = chat_template.join(JoinStrategy.ANTHROPIC, rendered)
# Output (double newline between messages):
# system: You are a helpful weather assistant.
#
# user: What's the weather in Tokyo?
```

## Output Parsers

### JSON Model Parser

Parse JSON into validated Pydantic models:

```python
parser = JsonModelParser[WeatherInfo](WeatherInfo, strict=True)
result = await parser.parse('{"temperature": 72.5, "condition": "sunny", "location": "NYC"}')
# Returns: WeatherInfo(temperature=72.5, condition='sunny', location='NYC')
```

### As-Is Parser

Return text unchanged:

```python
parser = AsIsParser()
result = await parser.parse("raw text")
# Returns: "raw text"
```

### Delimited Parser

Split text by delimiter:

```python
parser = DelimitedParser(sep="|")
result = await parser.parse("value1 | value2 | value3")
# Returns: {"0": "value1", "1": "value2", "2": "value3"}
```

## Variable Validation

Validate template variables before rendering:

```python
from pydantic_flow.prompt import collect_variables, validate_missing_or_extra

# Collect variables from template
vars_required = collect_variables(
    "Hello {name}, you are {age} years old!",
    TemplateFormat.F_STRING
)
# Returns: {"name", "age"}

# Validate provided variables
validate_missing_or_extra(
    vars_required={"name", "age"},
    provided={"name": "Alice", "age": 30, "extra": "ignored"}
)
# Raises VariableValidationError if missing or extra variables
```

## Integration with Flows

Use TypedPromptNode for workflow integration:

```python
from pydantic_flow.prompt import TypedPromptNode

node = TypedPromptNode(
    template=chat_template,
    parser=JsonModelParser(WeatherInfo),
    name="weather_query",
)

# In a flow context (once model adapters are implemented)
result = await node.run(QueryInput(location="London"))
```

## Observability

Templates include OpenTelemetry tracing:

```python
from pydantic_flow.prompt.serde import render_with_observability

# Automatic spans with:
# - Template format and hash
# - Variable count and names
# - Missing variables
# - Render time
# - Result length
```

## Best Practices

1. **Use Pydantic models for all inputs** - enforces type safety and validation
2. **Pair prompts with parsers** - structured output from the start
3. **Choose the right engine**:
   - F-string: Simple, fast, great for most use cases
   - Jinja2: Complex logic, filters, control flow
   - Mustache: Simplest, logic-less
4. **Validate variables** - catch missing variables early
5. **Use chat templates** - natural fit for LLM interactions

## Type Safety

Full type safety with IDE support:

```python
# Input and output types are tracked
template = PromptTemplate[QueryInput, WeatherInfo](...)

# Type checker knows the types
query: QueryInput = QueryInput(location="Paris")
result: WeatherInfo = await template.render_and_parse(query)

# Compile-time error if types don't match
wrong_query: dict = {"location": "Paris"}  # Error!
result = template.render(wrong_query)
```

## Advanced Examples

### Custom Parser

```python
class CustomParser(OutputParser[MyType]):
    async def parse(self, text: str) -> MyType:
        # Custom parsing logic
        return MyType.from_text(text)
```

### Dynamic Template Selection

```python
def get_template(format: TemplateFormat) -> PromptTemplate:
    return from_template(
        template="Hello {name}!",
        format=format,
        input_model=QueryInput,
    )
```

### Error Handling

```python
try:
    result = template.render(input_data)
except MissingVariableError as e:
    print(f"Missing variable: {e.variable}")
except TypeError as e:
    print(f"Invalid template type: {e}")
```
