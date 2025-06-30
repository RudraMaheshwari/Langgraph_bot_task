from typing import List
from langchain_core.messages import BaseMessage
from langchain.tools import tool
from src.models.llm_config import get_llm
from src.utils.message_filters import format_conversation_history
from src.agentic_prompts.interest_extraction_prompt import extract_interest_prompt

@tool
def extract_interests(messages: List[BaseMessage]) -> List[str]:
    """Extract student interests from conversation history using structured prompt."""
    try:
        llm = get_llm()
        chat_history = format_conversation_history(messages)
        prompt = extract_interest_prompt.format(chat_history=chat_history)

        response = llm.invoke(prompt)
        interests_text = response.content.strip().lower()

        if "no clear interests yet" in interests_text or not interests_text:
            return []

        interests = [i.strip() for i in interests_text.split(",")]

        filtered_interests = []
        for interest in interests:
            if len(interest) <= 2:
                continue
            if any(phrase in interest for phrase in [
                "based on the conversation",
                "there are no clear interests",
                "which does not reveal",
                "therefore",
                "no clear interests yet",
                "student has only said",
                "not enough information",
                "appropriate response"
            ]):
                continue
            filtered_interests.append(interest)

        return filtered_interests

    except Exception as e:
        print(f"[ERROR] extract_interests tool: {e}")
        return []
