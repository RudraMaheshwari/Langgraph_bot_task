from typing import List
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from src.models.llm_config import get_llm
from src.utils.message_filters import format_conversation_history
from src.agentic_prompts.interest_extraction_prompt import extract_interest_prompt

@tool
def extract_interests(messages: List[BaseMessage]) -> List[str]:
    """Extract student interests from conversation history using structured prompt."""
    llm = get_llm()
    
    chat_history = format_conversation_history(messages)
    prompt = extract_interest_prompt.format(chat_history=chat_history)

    try:
        response = llm.invoke(prompt)
        interests_text = response.content.strip().lower()

        if interests_text == "no clear interests yet." or not interests_text:
            return []

        interests = [interest.strip() for interest in interests_text.split(",")]
        return [interest for interest in interests if interest and len(interest) > 2]

    except Exception as e:
        print(f"Error extracting interests: {e}")
        return []
