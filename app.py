import os
import json
import datetime
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.utils.data_loader import load_course_data
from src.tools.course_retriever import CourseRetriever
from src.agent.course_agent import CourseRecommenderAgent
from src.agent.graph import create_course_recommender_graph
from src.schema.state import CourseRecommenderState

load_dotenv()
app = Flask(__name__)

FAISS_INDEX_PATH = "./faiss_store/index"

if os.path.exists(FAISS_INDEX_PATH):
    print("[INFO] Skipping course JSON load. Using existing FAISS index.")
    docs = []
else:
    print("[INFO] FAISS index not found. Loading course data from JSON.")
    docs = load_course_data()

course_retriever = CourseRetriever(docs)
course_agent = CourseRecommenderAgent(course_retriever)
conversation_graph = create_course_recommender_graph(course_agent)

user_sessions = {}

def get_user_id():
    """Simulate getting a unique user ID."""
    return "user_001"

def get_user_session(user_id: str) -> CourseRecommenderState:
    """Retrieve or initialize the user session."""
    if user_id not in user_sessions:
        user_sessions[user_id] = CourseRecommenderState(
            messages=[],
            grade=None,
            interests=[],
            credit_preference="any",
            conversation_stage="greeting",
            interest_turns=0,
            has_offered_recommendation=False,
            next_action=None,
            agent_scratchpad="",
            retrieved_courses=[],
            last_recommendation=None
        )
    return user_sessions[user_id]

def save_session(user_id: str, state: CourseRecommenderState):
    """Persist session in memory."""
    user_sessions[user_id] = state

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/set_grade", methods=["POST"])
def set_grade():
    data = request.get_json()
    if not data or 'grade' not in data:
        return jsonify({"error": "Grade is required"}), 400

    try:
        grade = int(data.get("grade"))
        if not 8 <= grade <= 12:
            return jsonify({"error": "Grade must be between 8 and 12"}), 400
    except ValueError:
        return jsonify({"error": "Grade must be a valid number"}), 400

    user_id = get_user_id()
    state = get_user_session(user_id)
    state["grade"] = grade
    save_session(user_id, state)
    
    return jsonify({"message": f"Grade set to {grade}"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    if not message:
        return jsonify({"error": "Message is required"}), 400

    user_id = get_user_id()
    state = get_user_session(user_id)

    if not state.get("grade"):
        return jsonify({"response": "Please set your grade first."}), 400

    state["messages"].append(HumanMessage(content=message))

    if "credit_type" in data:
        state["credit_preference"] = data["credit_type"]

    try:
        result = conversation_graph.invoke(state)
        save_session(user_id, result)

        ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
        if ai_messages:
            response = ai_messages[-1].content
        else:
            response = "I'm here to help you explore courses! What subjects interest you?"
        
        return jsonify({"response": response})
        
    except Exception as e:
        print(f"Error in chat processing: {e}")
        return jsonify({
            "response": "I'm having some trouble right now. Could you tell me about your interests or what subjects you enjoy?"
        }), 500

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    user_id = get_user_id()
    state = get_user_session(user_id)

    messages = []
    for msg in state.get("messages", []):
        messages.append({
            "role": "user" if msg.type == "human" else "bot",
            "content": msg.content
        })

    return jsonify({
        "messages": messages,
        "total_messages": len(messages)
    })

@app.route("/clear_history", methods=["POST"])
def clear_history():
    user_id = get_user_id()
    if user_id in user_sessions:
        del user_sessions[user_id]
    return jsonify({"message": "Chat history cleared successfully"})

@app.route("/get_user_info", methods=["GET"])
def get_user_info():
    user_id = get_user_id()
    state = get_user_session(user_id)

    return jsonify({
        "user_id": user_id,
        "grade": state.get("grade"),
        "interests": state.get("interests", []),
        "conversation_stage": state.get("conversation_stage"),
        "interest_turns": state.get("interest_turns", 0),
        "message_count": len(state.get("messages", [])),
        "has_offered_recommendation": state.get("has_offered_recommendation", False)
    })

@app.route("/save_chat_log", methods=["POST"])
def save_chat_log():
    user_id = get_user_id()
    state = get_user_session(user_id)

    messages = state.get("messages", [])
    if not messages:
        return jsonify({"error": "No chat history to save"}), 400

    chat_log = []
    for msg in messages:
        role = "user" if msg.type == "human" else "bot"
        chat_log.append({"role": role, "content": msg.content})

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_log_{user_id}_{timestamp}.json"
    save_path = os.path.join("chat_logs", filename)

    os.makedirs("chat_logs", exist_ok=True)

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(chat_log, f, ensure_ascii=False, indent=2)
        return jsonify({"message": f"Chat log saved successfully as {filename}"})
    except Exception as e:
        print(f"Error saving chat log: {e}")
        return jsonify({"error": "Failed to save chat log"}), 500

if __name__ == "__main__":
    app.run(debug=True)
