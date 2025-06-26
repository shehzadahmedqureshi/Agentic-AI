# Optional: Load environment variables if not using pipenv
# from dotenv import load_dotenv
# load_dotenv()

# === API Key Configuration ===
import os

GROQ_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_KEY = os.environ.get("TAVILY_API_KEY")

# === Language Model and Search Tool Setup ===
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

llm_instance = ChatGroq(model="llama-3.3-70b-versatile")
web_search = TavilySearchResults(max_results=2)

# === Agent Construction ===
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

DEFAULT_PROMPT = "You are a helpful and knowledgeable AI assistant."

def query_agent(model_name, messages, enable_search, prompt, provider):
    """
    Generate a response from the AI agent using the specified model and options.
    Args:
        model_name (str): The model identifier to use.
        messages (list): List of user messages.
        enable_search (bool): Whether to enable web search.
        prompt (str): System prompt for the agent.
        provider (str): Model provider name.
    Returns:
        str: The AI agent's response.
    """
    if provider == "Groq":
        model = ChatGroq(model=model_name)
    else:
        model = llm_instance  # fallback

    tools = [TavilySearchResults(max_results=2)] if enable_search else []
    agent = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=prompt or DEFAULT_PROMPT
    )
    state = {"messages": messages}
    result = agent.invoke(state)
    all_messages = result.get("messages", [])
    ai_responses = [msg.content for msg in all_messages if isinstance(msg, AIMessage)]
    return ai_responses[-1] if ai_responses else "No response generated."

