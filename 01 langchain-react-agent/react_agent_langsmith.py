# Complete LangChain ReAct Agent with Groq and LangSmith Tracking
# This script demonstrates building a ReAct agent that finds recent papers by researchers

import os
import sys
import getpass
from typing import Annotated
from dotenv import load_dotenv

# ============================================================================
# SETUP: Install required packages first
# ============================================================================
# pip install -U langgraph langchain-core langchain-groq python-dotenv

# ============================================================================
# STEP 1: Setup Environment Variables
# ============================================================================

# Load from .env file if it exists
load_dotenv()

def setup_environment():
    """Setup required environment variables for Groq, LangSmith, and LangChain."""
    
    # Setup Groq API Key
    if not os.environ.get("GROQ_API_KEY"):
        groq_key = getpass.getpass("Enter your Groq API Key (get it from https://console.groq.com): ")
        os.environ["GROQ_API_KEY"] = groq_key
    
    # Setup LangSmith (optional but recommended for tracing)
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("\n--- LangSmith Setup (Optional) ---")
        print("For better debugging and tracing, set up LangSmith:")
        print("1. Sign up at: https://smith.langchain.com")
        print("2. Create an API key in Settings > API Keys")
        
        use_langsmith = input("Do you want to enable LangSmith tracing? (y/n): ").lower()
        if use_langsmith == 'y':
            langsmith_key = getpass.getpass("Enter your LangSmith API Key: ")
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            
            # Optional: Set project name
            project_name = input("Enter project name (optional, press Enter for 'default'): ") or "default"
            os.environ["LANGCHAIN_PROJECT"] = project_name
    else:
        # If API key exists, enable tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# ============================================================================
# STEP 2: Import LangChain and LangGraph components
# ============================================================================

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# ============================================================================
# STEP 3: Define Tools for the Agent
# ============================================================================

@tool
def search_arxiv_by_author(author_name: str) -> str:
    """
    Search for recent papers by a specific author on ArXiv.
    This is a mock implementation - in production, integrate with ArXiv API.
    """
    # Mock implementation
    papers_db = {
        "Smith": [
            "Smith et al. (2025) - 'Advances in Climate Modeling' - ArXiv:2501.12345",
            "Smith, J. (2024) - 'Deep Learning for Environmental Data' - ArXiv:2412.54321",
            "Smith et al. (2024) - 'Neural Networks for Climate Prediction' - ArXiv:2411.11111"
        ],
        "Johnson": [
            "Johnson, K. (2025) - 'Machine Learning in Climate Science' - ArXiv:2502.99999",
            "Johnson et al. (2024) - 'Data Analysis Methods for Climate' - ArXiv:2412.77777"
        ],
        "Brown": [
            "Brown, L. (2025) - 'Climate Change Mitigation Strategies' - ArXiv:2501.55555"
        ]
    }
    
    author = author_name.title()
    if author in papers_db:
        papers = papers_db[author]
        return f"Found {len(papers)} papers by {author}:\n" + "\n".join(f"- {p}" for p in papers)
    else:
        return f"No papers found for author '{author_name}'. Try searching for: Smith, Johnson, or Brown"

@tool
def search_collaborators(author_name: str) -> str:
    """
    Find collaborators of a specific researcher.
    """
    collaborators_db = {
        "Smith": ["Johnson", "Williams", "Chen"],
        "Johnson": ["Smith", "Brown", "Martinez"],
        "Brown": ["Johnson", "Davis"],
        "Williams": ["Smith", "Garcia"],
        "Chen": ["Smith", "Anderson"]
    }
    
    author = author_name.title()
    if author in collaborators_db:
        collabs = collaborators_db[author]
        return f"Collaborators of {author}: {', '.join(collabs)}"
    else:
        return f"No collaborator information found for '{author_name}'"

@tool
def search_papers_by_keyword(keyword: str) -> str:
    """
    Search for papers related to a specific keyword in climate science.
    """
    keyword_lower = keyword.lower()
    papers_db = {
        "climate change": [
            "Smith et al. (2025) - 'Advances in Climate Modeling' - ArXiv:2501.12345",
            "Brown, L. (2025) - 'Climate Change Mitigation Strategies' - ArXiv:2501.55555",
            "Johnson, K. (2025) - 'Machine Learning in Climate Science' - ArXiv:2502.99999"
        ],
        "neural networks": [
            "Smith et al. (2024) - 'Neural Networks for Climate Prediction' - ArXiv:2411.11111",
            "Johnson et al. (2024) - 'Data Analysis Methods for Climate' - ArXiv:2412.77777"
        ],
        "deep learning": [
            "Smith, J. (2024) - 'Deep Learning for Environmental Data' - ArXiv:2412.54321"
        ]
    }
    
    results = []
    for key, papers in papers_db.items():
        if keyword_lower in key:
            results.extend(papers)
    
    if results:
        return f"Found {len(results)} papers about '{keyword}':\n" + "\n".join(f"- {p}" for p in results)
    else:
        return f"No papers found for keyword '{keyword}'. Try: climate change, neural networks, or deep learning"

# ============================================================================
# STEP 4: Create the ReAct Agent
# ============================================================================

def create_research_agent():
    """
    Create a ReAct agent that can research papers and collaborators.
    """
    # Initialize the Groq LLM (free tier available)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",  # Fast and capable free model
        temperature=0.7,
        max_tokens=1024
    )
    
    # Define tools for the agent
    tools = [
        search_arxiv_by_author,
        search_collaborators,
        search_papers_by_keyword
    ]
    
    # System prompt to guide the agent's behavior
    system_prompt = """You are a research assistant specialized in climate science and environmental research.
Your task is to help find recent papers and identify collaborators of researchers.

When asked about papers by a researcher's collaborators:
1. First find who collaborates with that researcher
2. Then search for papers by each collaborator
3. Compile and present the results clearly

Be systematic and thorough in your research."""
    
    # Create the agent using LangGraph's prebuilt create_react_agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    
    return agent

# ============================================================================
# STEP 5: Run the Agent
# ============================================================================

def run_agent_query(agent, query: str):
    """
    Execute a query through the agent and display results.
    """
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    try:
        # Invoke the agent with the user query
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        # Extract and display the final response
        final_message = result["messages"][-1]
        print("Agent Response:")
        print("-" * 70)
        print(final_message.content)
        print("-" * 70)
        
        return result
        
    except Exception as e:
        print(f"Error executing query: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# STEP 6: Main Execution
# ============================================================================

def main():
    """Main function to demonstrate the ReAct agent."""
    
    print("="*70)
    print("LangChain ReAct Agent with LangSmith Tracing")
    print("="*70)
    print("\nThis demo shows a ReAct agent that can:")
    print("- Search for papers by specific authors")
    print("- Find researcher collaborators")
    print("- Search papers by keywords")
    print("- Track execution with LangSmith (if configured)")
    
    # Setup environment
    print("\n1. Setting up environment variables...")
    setup_environment()
    
    if os.environ.get("LANGSMITH_API_KEY"):
        print(f"   ✓ LangSmith enabled! Project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
        print(f"   → Traces available at: https://smith.langchain.com")
    else:
        print("   ℹ LangSmith not configured (optional)")
    
    # Create the agent
    print("\n2. Creating ReAct agent with Groq LLM...")
    agent = create_research_agent()
    print("   ✓ Agent created successfully")
    
    # Example queries
    print("\n3. Running example queries...\n")
    
    queries = [
        "Find recent papers on climate change by Dr. Smith's collaborators.",
        "What are the latest papers about neural networks in climate science?",
        "Tell me about the researchers collaborating with Johnson on climate research."
    ]
    
    for query in queries:
        run_agent_query(agent, query)
        print("\n")
    
    # Interactive mode
    print("\n" + "="*70)
    print("Interactive Mode - Ask your own questions")
    print("(Type 'exit' to quit)")
    print("="*70)
    
    while True:
        user_query = input("\nEnter your query: ").strip()
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        
        if user_query:
            run_agent_query(agent, user_query)
    
    # Display LangSmith info if enabled
    if os.environ.get("LANGSMITH_API_KEY"):
        print("\n" + "="*70)
        print("LangSmith Traces")
        print("="*70)
        print(f"Project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
        print("View your traces at: https://smith.langchain.com")
        print("Check the traces to see:")
        print("- Agent reasoning steps (Thought/Action/Observation)")
        print("- Tool calls and their results")
        print("- Token usage and latency")
        print("- Complete execution flow")

if __name__ == "__main__":
    main()
