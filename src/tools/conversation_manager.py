from typing import List
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from src.models.llm_config import get_llm
from src.utils.message_filters import format_conversation_history
from src.agentic_prompts.interest_conversation_prompt import interest_conversation_prompt
from src.agentic_prompts.course_recommendation_prompt import recommendation_prompt

@tool
def generate_discovery_response(messages: List[BaseMessage], grade: int, interests: List[str]) -> str:
    """Generate a conversational discovery response using the interest_conversation_prompt."""
    llm = get_llm()
    chat_history = format_conversation_history(messages)
    last_user = messages[-1].content if messages and isinstance(messages[-1], BaseMessage) else ""

    prompt = interest_conversation_prompt.format(
        grade=grade,
        chat_history=chat_history,
        user_input=last_user
    )

    result = llm.invoke(prompt)
    return result.content.strip()

@tool
def generate_course_recommendation(
    query: str,
    grade: int,
    interests: List[str],
    credit_preference: str,
    course_context: str
) -> str:
    """Generate course recommendations using the recommendation_prompt."""
    llm = get_llm()
    interests_text = ", ".join(interests) if interests else "general academic interests"

    formatted = recommendation_prompt.format(
        context=course_context,
        grade=grade,
        interests=interests_text,
        credit_type=credit_preference,
        question=query
    )

    result = llm.invoke(formatted)
    return result.content.strip()
