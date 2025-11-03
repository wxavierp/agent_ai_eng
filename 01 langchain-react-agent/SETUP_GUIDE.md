# LangChain ReAct Agent with LangSmith Tracking - Complete Setup Guide

## Overview

This guide provides a complete working implementation of a LangChain ReAct agent that:
- Uses **Groq API** (free LLM provider) instead of OpenAI
- Tracks execution with **LangSmith** for observability
- Demonstrates tool use, reasoning, and action execution
- Includes interactive and programmatic usage

## Why Groq?

Groq is perfect for this tutorial because:
- ✓ **Completely Free** - No credit card required
- ✓ **Fast** - Optimized inference
- ✓ **Generous Rate Limits** - Suitable for experimentation
- ✓ **LangChain Integration** - Native support via `langchain-groq`

## Prerequisites

Before starting, ensure you have:
- Python 3.9 or higher
- pip (Python package manager)
- A terminal/command prompt

## Step 1: Get API Keys

### Groq API Key (Free)

1. Visit: https://console.groq.com
2. Sign up with email or Google/GitHub account
3. Go to **API Keys** section
4. Click **"Create New API Key"**
5. Copy and save the key securely

### LangSmith API Key (Optional but Recommended)

1. Visit: https://smith.langchain.com
2. Sign up (free account available)
3. Go to **Settings** → **API Keys**
4. Click **"Create API Key"**
5. Choose key type (Service Key recommended)
6. Copy and save the key

*Note: LangSmith free tier includes 5,000 traces/month with 14-day retention*

## Step 2: Installation

### Create a Project Directory

```bash
mkdir langchain-react-agent
cd langchain-react-agent
```

### Create Virtual Environment (Recommended)

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Install Required Packages

```bash
pip install -U langgraph langchain-core langchain-groq python-dotenv
```

**Package Breakdown:**
- `langgraph` - Graph-based agent orchestration framework
- `langchain-core` - Core LangChain abstractions
- `langchain-groq` - Groq integration for LangChain
- `python-dotenv` - Load environment variables from .env file

## Step 3: Environment Setup

### Option A: Using .env File (Recommended)

Create a `.env` file in your project directory:

```
# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here

# LangSmith Configuration (Optional)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=react-agent-demo
```

**Important:** Add `.env` to `.gitignore` to prevent accidentally committing API keys:

```
echo ".env" >> .gitignore
```

### Option B: Using Environment Variables

**On macOS/Linux:**

```bash
export GROQ_API_KEY="your_groq_api_key_here"
export LANGSMITH_API_KEY="your_langsmith_api_key_here"
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=react-agent-demo
```

**On Windows (PowerShell):**

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
$env:LANGSMITH_API_KEY="your_langsmith_api_key_here"
$env:LANGCHAIN_TRACING_V2="true"
$env:LANGCHAIN_PROJECT="react-agent-demo"
```

## Step 4: Run the Agent

```bash
python react_agent_langsmith.py
```

The script will:
1. Prompt for API keys (if not in environment)
2. Create a ReAct agent with Groq
3. Run example queries
4. Enter interactive mode

## Understanding the Code

### Key Components

#### 1. **Tools Definition**

Tools are functions the agent can call to gather information:

```python
@tool
def search_arxiv_by_author(author_name: str) -> str:
    """Search for recent papers by a specific author on ArXiv."""
    # Implementation
    return results
```

#### 2. **Agent Creation**

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,              # Groq LLM instance
    tools=tools,            # List of available tools
    prompt=system_prompt    # Instructions for the agent
)
```

#### 3. **Agent Invocation**

```python
result = agent.invoke({
    "messages": [HumanMessage(content="Your query here")]
})
```

### ReAct Pattern Flow

The ReAct (Reasoning and Acting) pattern works as follows:

```
User Query
    ↓
[Thought] - Agent reasons about what to do
    ↓
[Action] - Agent selects a tool to call
    ↓
[Observation] - Tool returns result
    ↓
[Loop] - Repeat until final answer reached
    ↓
Final Response to User
```

## LangSmith Integration

### Automatic Tracking

Once LangSmith API key is configured, all agent executions are automatically tracked:

```
Environment Variable: LANGSMITH_API_KEY
         ↓
LangChain detects and enables tracing
         ↓
Agent invocations are logged to LangSmith
         ↓
View traces at: https://smith.langchain.com
```

### What Gets Tracked

- **Traces** - Complete execution flow of your agent
- **Tokens** - Input and output token usage
- **Latency** - Response times for each step
- **Tool Calls** - Which tools were called and their results
- **Agent Reasoning** - Thought process and decision-making

### Viewing Traces

1. Log in to https://smith.langchain.com
2. Select your project (default: "default" or your custom name)
3. View all traces from recent runs
4. Click on any trace to see:
   - Input/output
   - Tool calls and results
   - Token usage
   - Execution timeline

## Example Usage

### Command-Line Usage

```bash
$ python quickstart.py
```

## Troubleshooting

### Issue: "Authentication failed for Groq"

**Solution:** Verify your GROQ_API_KEY is correct:
- Check for trailing spaces
- Ensure you copied the full key
- Regenerate the key if needed at https://console.groq.com

### Issue: "LANGSMITH_API_KEY not found"

**Solution:** LangSmith is optional. If you don't configure it:
- Agent will still work fine
- You just won't have execution traces
- To enable, set the environment variable

### Issue: "Tool not found" error

**Solution:** Ensure tool names match exactly:
- Check tool decorator: `@tool`
- Verify function name doesn't have typos
- Make sure tool is added to `tools` list

### Issue: "Rate limit exceeded"

**Solution:** Groq has generous free limits, but if exceeded:
- Wait a few minutes before retrying
- Reduce `temperature` for shorter responses
- Reduce `max_tokens` parameter

## Summary

You now have a fully functional ReAct agent that:
- ✓ Uses free Groq LLM API
- ✓ Performs multi-step reasoning
- ✓ Calls external tools
- ✓ Tracks execution with LangSmith
- ✓ Provides interactive interface

The agent demonstrates production-ready patterns for building autonomous AI systems!
