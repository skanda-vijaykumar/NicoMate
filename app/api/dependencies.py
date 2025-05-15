import logging
import threading
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor

from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Import these at the top to avoid E402 errors
# (With a comment explaining why they were previously imported later)
from app.core.data_loader import load_data, process_data
from app.services.tool_factory import create_tools

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    MAX_AGENTS,
    EXTRACTED_DATA_DIR,
)

# Global variables
vector_index_markdown = None
keyword_index_markdown = None
vector_index_markdown_lab = None
keyword_index_markdown_lab = None
startup_lock = threading.Lock()
startup_complete = threading.Event()
agent_pool = None
agent_lock = threading.Lock()
agent_queue = queue.Queue()


async def initialize_data_and_models():
    """Initialize data, models, and indices on startup."""
    global vector_index_markdown, keyword_index_markdown, vector_index_markdown_lab, keyword_index_markdown_lab
    global agent_pool

    with startup_lock:
        try:
            # Load and process data
            logging.info("Loading data...")
            documents1, documents3 = load_data(EXTRACTED_DATA_DIR)
            logging.info(
                f"Loaded {len(documents1)} catalog documents and {len(documents3)} lab documents"
            )

            logging.info("Processing data...")
            result_indices = process_data(documents1, documents3)

            if result_indices and len(result_indices) == 4:
                (
                    vector_index_markdown,
                    keyword_index_markdown,
                    vector_index_markdown_lab,
                    keyword_index_markdown_lab,
                ) = result_indices
                logging.info("Successfully loaded all indices")
            else:
                logging.warning(
                    f"Warning: Data processing returned {len(result_indices) if result_indices else 0} indices instead of 4"
                )

            # Initialize agent pool
            logging.info("Initializing agent pool...")
            agent_pool = ThreadPoolExecutor(max_workers=MAX_AGENTS)

            # Create tools
            tools = create_tools()

            # Pre-create agents
            for i in range(MAX_AGENTS):
                logging.info(f"Creating agent {i + 1}/{MAX_AGENTS}...")  # Added space around + operator
                agent = create_isolated_agent(tools)
                agent_queue.put(agent)

            logging.info(f"Initialized {MAX_AGENTS} agents in the pool")

            # Set completion event
            startup_complete.set()

        except Exception as e:
            logging.error(f"Error in data and model initialization: {str(e)}")
            # Set the event even on failure to avoid hanging requests
            startup_complete.set()
            # Raise the exception to mark initialization as failed
            raise


def create_isolated_agent(tools):
    """Create an isolated agent with its own LLM instance."""
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
        num_ctx=8152,
        cache=False,
        base_url=OLLAMA_BASE_URL,
        client_kwargs={"timeout": 60},
        client_id=f"agent-{uuid.uuid4()}",
    )
    prompt = hub.pull("intern/ask11")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executer = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=2,
        max_execution_time=None,
        callbacks=[StreamingStdOutCallbackHandler()],
        return_intermediate_steps=True,
        early_stopping_method="force",
    )
    return agent_executer


async def get_agent(tools):
    """Get an agent from the pool or create a new one."""
    try:
        # Try to get an existing agent from the queue
        agent = agent_queue.get_nowait()
        return agent
    except queue.Empty:
        # If no agents are available, create a new one
        return create_isolated_agent(tools)


def return_agent(agent):
    """Return an agent to the pool."""
    try:
        agent_queue.put(agent, block=False)
    except queue.Full:
        pass
