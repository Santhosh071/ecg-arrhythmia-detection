from langchain.memory import ConversationBufferWindowMemory


def get_short_term_memory(k: int = 10) -> ConversationBufferWindowMemory:
    """Return a fresh short-term conversation memory window."""
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )
