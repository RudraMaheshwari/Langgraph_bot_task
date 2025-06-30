from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from src.schema.state import CourseRecommenderState
from src.agent.course_agent import CourseRecommenderAgent
from src.tools.interest_extractor import extract_interests


def create_course_recommender_graph(course_agent: CourseRecommenderAgent) -> StateGraph:
    def process_user_input(state: CourseRecommenderState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        last_message = messages[-1].content.lower() if messages and isinstance(messages[-1], HumanMessage) else ""

        interests = state.get("interests", [])
        interest_turns = state.get("interest_turns", 0)
        stage = state.get("conversation_stage")

        if any(exit_phrase in last_message for exit_phrase in ["bye", "exit", "quit", "stop", "no thanks", "goodbye"]):
            return {**state, "conversation_stage": "complete"}
    
        if stage == "complete" and last_message:
            return {**state, "conversation_stage": "discovery", "interest_turns": 0}

        if stage == "prompt_recommendation":
            if any(affirm in last_message for affirm in ["yes", "yeah", "sure", "of course", "please",'yes if you have', 'yes for sure', 'ok go ahead','yes please','yep']):
                return {**state, "conversation_stage": "recommendation"}
            elif any(neg in last_message for neg in ["no", "not really", "maybe later",'not right now']):
                return {**state, "conversation_stage": "discovery"}
            else:
                return state

        if interests and len(interests) >= 2:
            if not state.get("has_prompted_recommendation", False):
                return {**state, "conversation_stage": "prompt_recommendation"}
            return {**state, "conversation_stage": "recommendation"}

        if interest_turns >= 3:
            return {**state, "conversation_stage": "prompt_recommendation"}

        return {**state, "conversation_stage": "discovery"}


    def extract_user_interests(state: CourseRecommenderState) -> Dict[str, Any]:
        try:
            messages = state.get("messages", [])
            new_interests = extract_interests.invoke({"messages": messages})
            if not isinstance(new_interests, list):
                print("[WARN] extract_interests returned non-list result. Skipping update.")
                return state
            current_interests = state.get("interests", [])
            all_interests = list(dict.fromkeys(current_interests + new_interests))
            return {**state, "interests": all_interests}
        except Exception as e:
            print(f"[ERROR] Failed to extract user interests: {e}")
            return state

    def generate_response(state: CourseRecommenderState) -> Dict[str, Any]:
        try:
            stage = state.get("conversation_stage")
            messages = state.get("messages", [])

            if stage == "prompt_recommendation":
                prompt = f"Would you like me to recommend some courses related to {', '.join(state.get('interests', []))}?"
                return {
                    **state,
                    "messages": messages + [AIMessage(content=prompt)],
                    "has_prompted_recommendation": True
                }

            result = course_agent.process_message(state)
            response = result.get("response", "").strip() or "Could you tell me more about what you like?"

            updated_messages = messages + [AIMessage(content=response)]
            interest_turns = state.get("interest_turns", 0)
            if stage == "discovery":
                interest_turns += 1

            new_state = {
                **state,
                "messages": updated_messages,
                "interest_turns": interest_turns
            }

            if stage == "recommendation":
                new_state.update({
                    "last_recommendation": response,
                    "has_offered_recommendation": True,
                    "conversation_stage": "complete"
                })

            print(f"[DEBUG] Stage before update: {stage}")
            print(f"[DEBUG] Interests: {state.get('interests')}")
            return new_state

        except Exception as e:
            print(f"[ERROR] Response generation failed: {e}")
            fallback = "Hmm, I'm still getting to know you. What else do you enjoy?"
            return {
                **state,
                "messages": state.get("messages", []) + [AIMessage(content=fallback)]
            }

    def should_continue(state: CourseRecommenderState) -> str:
        return "complete"

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

    return workflow.compile()
