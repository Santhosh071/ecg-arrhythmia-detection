import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from .llm_config   import get_llm, check_llm_status
from .memory       import ECGDatabase, get_short_term_memory
from .agent_tools  import ALL_TOOLS, init_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = PromptTemplate.from_template("""
You are an AI-assisted ECG arrhythmia monitoring assistant for clinical decision support.

You have access to the following tools:
{tools}

Tool names: {tool_names}

STRICT RULES:
1. You are a DECISION SUPPORT tool only. Never make final clinical decisions alone.
2. Always include a disclaimer when providing clinical interpretations.
3. Use tools to answer — do not guess from memory alone.
4. If the question is outside cardiology/ECG scope, politely decline.

Use this format:
Thought: what do I need to do?
Action: tool_name
Action Input: {{"key": "value"}}
Observation: tool result
... (repeat as needed)
Thought: I now have enough to answer
Final Answer: your complete response

Patient context: {chat_history}

Question: {input}
{agent_scratchpad}
""")

class ECGAgent:
    """
    Main agent class. Wraps LangChain AgentExecutor with ECG-specific tools.

    Attributes
    ----------
    db       : ECGDatabase — long-term SQLite memory
    memory   : ConversationBufferWindowMemory — short-term window
    executor : LangChain AgentExecutor
    ready    : bool
    """
    def __init__(self, db_path: str = None):
        self.db       = ECGDatabase(db_path) if db_path else ECGDatabase()
        self.memory   = get_short_term_memory(k=10)
        self.executor = None
        self.ready    = False

    def start(self) -> "ECGAgent":
        """
        Initialise LLM, inject into tools, build agent executor.
        Call once at application startup.

        Returns self for chaining: agent = ECGAgent().start()

        Raises
        ------
        ConnectionError : if Groq API unreachable
        ValueError      : if GROQ_API_KEY missing
        """
        status = check_llm_status()
        if not status["available"]:
            if not status["has_key"]:
                raise ValueError("GROQ_API_KEY missing in .env")
            raise ConnectionError("Groq API unreachable. Check internet connection.")
        llm = get_llm(temperature=0.2)
        init_tools(self.db, llm)
        agent = create_react_agent(
            llm     = llm,
            tools   = ALL_TOOLS,
            prompt  = SYSTEM_PROMPT,
        )
        self.executor = AgentExecutor(
            agent          = agent,
            tools          = ALL_TOOLS,
            memory         = self.memory,
            verbose        = False,
            max_iterations = 5,
            handle_parsing_errors = True,
        )
        self.ready = True
        logger.info("[agent] ECGAgent ready.")
        return self

    def run(self, user_input: str, patient_id: str = None) -> str:
        """
        Run the agent on a user query.

        Parameters
        ----------
        user_input : str — clinician's question or command
        patient_id : str — optional, appended to context if provided

        Returns
        -------
        Agent response string

        Raises
        ------
        RuntimeError : if agent not started
        """
        if not self.ready:
            raise RuntimeError("Call ECGAgent.start() before run().")
        context = user_input
        if patient_id:
            context = f"[Patient: {patient_id}] {user_input}"
        try:
            result = self.executor.invoke({"input": context})
            output = result.get("output", "No response generated.")
            self.db.log_tool_call("agent_run", context[:200], output[:300], patient_id)
            return output
        except Exception as e:
            logger.error(f"[agent] run error: {e}")
            return f"Agent error: {e}"

    def run_tool_directly(self, tool_name: str, input_json: str) -> str:
        """
        Call a specific tool directly without going through the agent LLM.
        Faster for programmatic use (e.g. dashboard calling assess_risk directly).

        Parameters
        ----------
        tool_name  : one of the 8 tool names as a string
        input_json : JSON string matching the tool's expected input

        Returns
        -------
        Tool output string
        """
        tool_map = {t.name: t for t in ALL_TOOLS}
        if tool_name not in tool_map:
            return f"Unknown tool: {tool_name}. Available: {list(tool_map.keys())}"
        try:
            return tool_map[tool_name].run(input_json)
        except Exception as e:
            logger.error(f"[agent] direct tool error {tool_name}: {e}")
            return f"Tool error ({tool_name}): {e}"

    def reset_memory(self):
        """Clear short-term conversation memory (start fresh session)."""
        self.memory.clear()
        logger.info("[agent] Short-term memory cleared.")

    def status(self) -> dict:
        """Return agent status dict for dashboard."""
        llm_status = check_llm_status()
        return {
            "agent_ready" : self.ready,
            "llm_status"  : llm_status,
            "tools_count" : len(ALL_TOOLS),
            "tool_names"  : [t.name for t in ALL_TOOLS],
            "memory_type" : "ConversationBufferWindowMemory(k=10)",
            "db_type"     : "SQLite",
        }