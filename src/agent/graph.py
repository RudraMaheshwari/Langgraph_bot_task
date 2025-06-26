from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from src.schema.state import CourseRecommenderState
from src.agent.course_agent import CourseRecommenderAgent
from src.tools.interest_extractor import extract_interests

def create_course_recommender_graph(course_agent: CourseRecommenderAgent) -> StateGraph:
    """Create the course recommender conversation graph with in-memory checkpointing."""

    def process_user_input(state: CourseRecommenderState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], HumanMessage):
            return state

        interests = state.get("interests", [])
        interest_turns = state.get("interest_turns", 0)

        if (interests and len(interests) >= 2) or interest_turns >= 3:
            return {**state, "conversation_stage": "recommendation"}

        return {**state, "conversation_stage": "discovery"}

    def extract_user_interests(state: CourseRecommenderState) -> Dict[str, Any]:
        try:
            messages = state.get("messages", [])
            new_interests = extract_interests.invoke({"messages": messages})

            current_interests = state.get("interests", [])
            all_interests = list(dict.fromkeys(current_interests + new_interests))

            return {**state, "interests": all_interests}
        except Exception as e:
            print(f"Error extracting interests: {e}")
            return state

    def generate_response(state: CourseRecommenderState) -> Dict[str, Any]:
        try:
            result = course_agent.process_message(state)
            response = result.get("response", "").strip()

            if not isinstance(response, str) or not response:
                response = "I'm still thinking. Could you share more about what you're interested in?"

            updated_messages = state.get("messages", []) + [AIMessage(content=response)]
            interest_turns = state.get("interest_turns", 0)

            if state.get("conversation_stage") == "discovery":
                interest_turns += 1

            if state.get("conversation_stage") == "recommendation":
                return {
                    **state,
                    "messages": updated_messages,
                    "interest_turns": interest_turns,
                    "last_recommendation": response,
                    "has_offered_recommendation": True,
                    "conversation_stage": "complete"
                }

            return {
                **state,
                "messages": updated_messages,
                "interest_turns": interest_turns
            }

        except Exception as e:
            print(f"Error generating response: {e}")
            fallback = "Hmm, I'm still getting to know you. What else do you enjoy?"
            updated_messages = state.get("messages", []) + [AIMessage(content=fallback)]
            return {**state, "messages": updated_messages}

    def should_continue(state: CourseRecommenderState) -> str:
        if state.get("conversation_stage") == "complete":
            return "complete"
        return "continue"

    workflow = StateGraph(CourseRecommenderState)
    workflow.add_node("process_input", process_user_input)
    workflow.add_node("extract_interests", extract_user_interests)
    workflow.add_node("generate_response", generate_response)

    workflow.set_entry_point("process_input")
    workflow.add_edge("process_input", "extract_interests")
    workflow.add_edge("extract_interests", "generate_response")
    workflow.add_conditional_edges("generate_response", should_continue, {
        "continue": "process_input",
        "complete": END
    })

    memory_saver = MemorySaver()
    return workflow.compile()
    