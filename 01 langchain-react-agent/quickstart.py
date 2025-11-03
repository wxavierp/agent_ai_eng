# Quick Start - Minimal Working Example
# This is the simplest version to get running immediately

import os
from dotenv import load_dotenv

# SETUP: pip install langgraph langchain-core langchain-groq python-dotenv

load_dotenv()

# Set API keys or input them
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = input("Enter Groq API Key: ")

# Enable LangSmith (optional - just set the env var if you have a key)
# os.environ["LANGSMITH_API_KEY"] = "your_key_here"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Import components
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# ============================================================================
# STEP 1: Define Tools
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    # Mock data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 65°F",
        "London": "Rainy, 55°F",
        "Tokyo": "Sunny, 75°F"
    }
    return weather_data.get(city, f"Weather data for {city} not available")

@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    population_data = {
        "San Francisco": "873,965",
        "New York": "8,335,897",
        "London": "9,002,488",
        "Tokyo": "37,400,068"
    }
    return population_data.get(city, f"Population data for {city} not available")

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Mock search results
    return f"Search results for '{query}': Found 1.2M results (mocked)"

# ============================================================================
# STEP 2: Create Agent
# ============================================================================

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

# Create agent with tools
agent = create_react_agent(
    model=llm,
    tools=[get_weather, get_population, search_web],
    prompt="You are a helpful assistant that provides information about cities."
)

# ============================================================================
# STEP 3: Run Agent
# ============================================================================

# Example 1: Single query
print("="*60)
print("EXAMPLE 1: Basic Query")
print("="*60)

result = agent.invoke({
    "messages": [HumanMessage(content="What's the weather in San Francisco?")]
})

print(result["messages"][-1].content)

# Example 2: Complex query requiring multiple tools
print("\n" + "="*60)
print("EXAMPLE 2: Multi-Tool Query")
print("="*60)

result = agent.invoke({
    "messages": [HumanMessage(content="Tell me about New York - what's the weather and population?")]
})

print(result["messages"][-1].content)

# Example 3: Interactive mode
print("\n" + "="*60)
print("EXAMPLE 3: Interactive Mode")
print("="*60)
print("Type 'exit' to quit\n")

while True:
    query = input("Ask about cities: ").strip()
    if query.lower() == "exit":
        break
    
    if query:
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        print(f"\nAgent: {result['messages'][-1].content}\n")

print("\nDone!")
