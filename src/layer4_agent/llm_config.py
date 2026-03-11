import os
import socket
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_HOST      = "api.groq.com"
GROQ_TIMEOUT   = 5

def is_internet_available() -> bool:
    """Check if Groq API is reachable."""
    try:
        socket.setdefaulttimeout(GROQ_TIMEOUT)
        socket.create_connection((GROQ_HOST, 443))
        return True
    except OSError:
        return False

def get_llm(temperature: float = 0.2) -> ChatGroq:
    """
    Return a ChatGroq LLM instance.

    Parameters
    ----------
    temperature : float — 0.0 = deterministic, 1.0 = creative
                  0.2 is good for clinical explanations

    Raises
    ------
    ConnectionError : if Groq API is unreachable
    ValueError      : if GROQ_API_KEY is missing from .env
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "PASTE_YOUR_NEW_GROQ_KEY_HERE":
        raise ValueError(
            "GROQ_API_KEY missing in .env\n"
            "Get a free key at https://console.groq.com"
        )
    if not is_internet_available():
        raise ConnectionError(
            "Cannot reach Groq API (api.groq.com).\n"
            "Check your internet connection and try again."
        )
    return ChatGroq(
        api_key     = GROQ_API_KEY,
        model_name  = GROQ_MODEL,
        temperature = temperature,
    )

def check_llm_status() -> dict:
    """Return LLM availability status — used by dashboard and agent startup."""
    internet = is_internet_available()
    has_key  = bool(GROQ_API_KEY and GROQ_API_KEY != "PASTE_YOUR_NEW_GROQ_KEY_HERE")
    return {
        "available"   : internet and has_key,
        "internet"    : internet,
        "has_key"     : has_key,
        "model"       : GROQ_MODEL,
        "provider"    : "Groq",
    }