from typing import TypedDict, List, Optional, Annotated
from typing_extensions import Literal
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class CourseRecommenderState(TypedDict):
    """State schema for the course recommender agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    grade: Optional[int]
    interests: List[str]
    credit_preference: Optional[str]

    conversation_stage: Literal["greeting", "discovery", "recommendation", "complete"]
    interest_turns: int
    has_offered_recommendation: bool

    next_action: Optional[str]
    agent_scratchpad: str

    retrieved_courses: List[dict]
    last_recommendation: Optional[str]

def reduce_interests(left: List[str], right: List[str]) -> List[str]:
    """Reducer for interests to avoid duplicates."""
    combined = left + right
    return list(dict.fromkeys(combined))
