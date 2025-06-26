from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def filter_recent_messages(messages: List[BaseMessage], max_messages: int = 10) -> List[BaseMessage]:
    return messages[-max_messages:] if len(messages) > max_messages else messages

def filter_by_message_type(messages: List[BaseMessage], include_types: List[str]) -> List[BaseMessage]:
    filtered = []
    for msg in messages:
        if msg.type in include_types:
            filtered.append(msg)
    return filtered

def format_conversation_history(messages: List[BaseMessage], max_exchanges: int = 5) -> str:
    recent_messages = filter_recent_messages(messages, max_exchanges * 2)
    
    formatted = ""
    for msg in recent_messages:
        role = "Student" if isinstance(msg, HumanMessage) else "Bot"
        formatted += f"{role}: {msg.content}\n"
    
    return formatted.strip()