import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st
from streamlit_chat import message
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from KnowledgeBase import KnowledgeBase
from HistoryAdding import HistoryAdder
from Retrievial import Retriever
from Generation import Generator

# Load environment variables
load_dotenv()

# File to store chat histories
CHAT_HISTORY_FILE = "chat_histories.json"

# Initialize components
def init_components():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="AIzaSyBYNcmgcf3vOTBpsU7IOvGFofzK5hDkV4A")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    ha = HistoryAdder(llm)
    kb = KnowledgeBase(embeddings, "knowledge_base")
    db = kb.modify_vector_store()
    r = Retriever(db, "similarity", {"k": 2})
    
    question_generator_prompt = """
        You are an AI-powered TOEIC question generator designed to create high-quality TOEIC-style questions focus on Writing task based on a given knowledge base. Your goal is to generate questions that test students' English proficiency in alignment with TOEIC standards.

        Instructions:
            Input Knowledge Base: You will receive a knowledge base that contains grammar rules, vocabulary lists, question examples, format of the exam. Use this information as the foundation for generating questions.
            Question Types: Generate questions that reflect real TOEIC exam formats style based on given knowledge base.
            Question Format: You MUST point out which part of the exam the question belongs to and direction of how to answer the questions.
            Difficulty Level: Adjust the difficulty of the questions based on the knowledge base provided. Label each question as Beginner, Intermediate, or Advanced.
        
        Here is the knowledge base: 
            {kb}
    """
    
    grader_prompt = """
        You are a TOEIC teacher who specializes in grading students' answers to writing questions. Your objective is to evaluate a student's written response using a set of clearly defined criteria. For every essay you grade, please follow these steps:
            1. Task Achievement / Task Response:
                _ Check if the response fully addresses the prompt and covers all required parts.
                _ Assess whether the content is relevant, well-developed, and supported with details or examples.

            2. Coherence and Cohesion:
                _ Evaluate the overall organization and structure of the response.
                _ Look for clear paragraphing, logical sequencing of ideas, and the effective use of linking devices (e.g., transition words, referencing).

            3. Lexical Resource:
                _ Assess the range and accuracy of vocabulary.
                _ Check for appropriate word choice, collocations, and correct spelling.

            4. Grammatical Range and Accuracy:
                _ Evaluate the variety and complexity of sentence structures (including simple, compound, and complex sentences).
                _ Ensure grammar, punctuation, and syntax are used correctly with minimal errors.

            5. Task-Specific Considerations:
                _ Verify that the response follows the required format, tone, and style (e.g., essay, report, letter) as specified by the question.

            For each criterion, provide a qualitative score (or descriptive feedback, out of 10 score) and specific comments on strengths and areas for improvement. Then, calculate an overall score by averaging the scores from the four main criteria (Task Achievement/Response, Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy).
            You MUST not afraid to give a low grade.
            Just give your evaluation and comment, dont add anythings else.
            In your final output, include:
                _ A brief overall summary of the response's performance.
                _ Detailed feedback for each criterion.
                _ A final overall score along with recommendations for improvement.
                _ Make sure your evaluation is objective, evidence-based, and strictly focused on the content and language of the essay.
    """
    
    question_agent = Generator(llm, question_generator_prompt)
    grade_agent = Generator(llm, grader_prompt)
    
    return llm, ha, r, question_agent, grade_agent

# Helper function to convert langchain Message objects to serializable format
def serialize_message(message_obj):
    if isinstance(message_obj, AIMessage):
        return {
            "type": "AIMessage",
            "content": message_obj.content
        }
    elif isinstance(message_obj, HumanMessage):
        return {
            "type": "HumanMessage",
            "content": message_obj.content
        }
    else:
        return str(message_obj)  # Fallback for unknown types

# Helper function to deserialize messages
def deserialize_message(message_dict):
    if not isinstance(message_dict, dict):
        return message_dict
    
    message_type = message_dict.get("type")
    content = message_dict.get("content")
    
    if message_type == "AIMessage":
        return AIMessage(content=content)
    elif message_type == "HumanMessage":
        return HumanMessage(content=content)
    else:
        return message_dict

# Session management functions
def get_session_id():
    if "session_id" not in st.query_params:
        new_session_id = str(uuid.uuid4())
        st.query_params["session_id"] = new_session_id
        save_session_info(new_session_id)
    else:
        histories = load_chat_histories()
        if st.query_params["session_id"] not in histories:
            save_session_info(st.query_params["session_id"])
    return st.query_params["session_id"]

def get_session_url(session_id):
    base_url = st.get_option("server.baseUrlPath") or ""
    return f"{base_url}?session_id={session_id}"

def save_session_info(session_id):
    histories = load_chat_histories()
    histories[session_id] = {
        "created_at": datetime.now().isoformat(),
        "messages": [],
        "history_for_context": []
    }
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(histories, f, indent=2)

def load_chat_histories():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            try:
                histories = json.load(f)
                for session_id, session_data in histories.items():
                    if not isinstance(session_data, dict) or "created_at" not in session_data:
                        histories[session_id] = {
                            "messages": session_data if isinstance(session_data, list) else [],
                            "created_at": datetime.now().isoformat(),
                            "history_for_context": []
                        }
                return histories
            except json.JSONDecodeError:
                st.error("Error loading chat histories. Starting with a clean slate.")
                return {}
    return {}

def load_chat_history(session_id):
    histories = load_chat_histories()
    if session_id in histories:
        session_data = histories[session_id]
        if isinstance(session_data, dict):
            st.session_state.messages = session_data.get("messages", [])
            
            # Deserialize history_for_context messages
            serialized_history = session_data.get("history_for_context", [])
            st.session_state.hist_for_context = [deserialize_message(msg) for msg in serialized_history]
        else:
            st.session_state.messages = session_data
            st.session_state.hist_for_context = []
        return True
    return False

def add_message(role, content):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    save_chat_history()

def save_chat_history():
    histories = load_chat_histories()
    current_session = histories.get(st.query_params["session_id"], {})
    current_session["messages"] = st.session_state.messages
    
    # Serialize the langchain Message objects
    current_session["history_for_context"] = [serialize_message(msg) for msg in st.session_state.hist_for_context]
    
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(histories, f, indent=2)

def clear_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)
    st.session_state.messages = []
    st.session_state.hist_for_context = []
    save_session_info(st.query_params["session_id"])

def create_new_session():
    new_session_id = str(uuid.uuid4())
    save_session_info(new_session_id)
    new_session_url = get_session_url(new_session_id)
    st.markdown(f'<meta http-equiv="refresh" content="0;url={new_session_url}">', unsafe_allow_html=True)
    st.stop()

def truncate_session_id(session_id):
    return f"{session_id[:6]}..."

def format_datetime(iso_string):
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "Unknown date"

def main():
    st.set_page_config(page_title="TOEIC Practice Assistant", page_icon="📝", layout="wide")
    st.title("📝 TOEIC Practice Assistant")
    
    # Initialize session state
    if "disable_input" not in st.session_state:
        st.session_state.disable_input = False
    
    if "hist_for_context" not in st.session_state:
        st.session_state.hist_for_context = []
    
    if "question_mode" not in st.session_state:
        st.session_state.question_mode = "practice"  # practice or test
    
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    
    if "answer_submitted" not in st.session_state:
        st.session_state.answer_submitted = False
    
    # Initialize components
    llm, ha, r, question_agent, grade_agent = init_components()
    
    # Get session ID
    session_id = get_session_id()
    if "messages" not in st.session_state:
        if not load_chat_history(session_id):
            st.session_state.messages = []
    
    truncated_session_id = truncate_session_id(session_id)
    st.info(f"Current Session ID: {truncated_session_id}")
    
    # Create sidebar
    with st.sidebar:
        st.title("Session Management")
        
        if st.button("Create New Session"):
            create_new_session()
        
        if st.button("Clear All Chat Histories"):
            clear_chat_history()
            st.rerun()
        
        st.subheader("Previous Sessions")
        histories = load_chat_histories()
        for old_session_id, session_info in histories.items():
            if old_session_id != session_id:
                if isinstance(session_info, dict) and "created_at" in session_info:
                    created_at = format_datetime(session_info["created_at"])
                else:
                    created_at = "Unknown date"
                session_url = get_session_url(old_session_id)
                if st.button(f"Session from {created_at}", key=f"button_{created_at}_{old_session_id}"):
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={session_url}">', unsafe_allow_html=True)
                    st.stop()
        
        st.subheader("Mode Selection")
        question_type = st.selectbox(
            "Select Question Type",
            ["Writing Part 1", "Writing Part 2", "Writing Part 3"],
            key="question_type"
        )
    
    # Main area
    chat_container = st.container(height=500, border=True)
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            avatar_style = 'bottts-neutral' if msg["role"] != "user" else None
            message(msg["content"], is_user=msg["role"] == "user", key=f"{i}_{msg['role']}", avatar_style=avatar_style)
    
    # Question generation area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Generate New Question"):
            st.session_state.current_question = None
            st.session_state.answer_submitted = False
            
            question_prompt = f"Generate a TOEIC writing part {question_type.split(' ')[-1]} question"
            print(question_prompt)
            rel_docs = r.get_context(question_prompt)
            
            question = question_agent.generate_answer(question_prompt, rel_docs)
            st.session_state.current_question = question
            
            with chat_container:
                message(question, is_user=False, key=f"question_{len(st.session_state.messages)}", avatar_style='bottts-neutral')
            
            add_message("assistant", question)
            st.session_state.hist_for_context.append(AIMessage(content=question))
    
    # Answer area
    if st.session_state.current_question and not st.session_state.answer_submitted:
        with col2:
            st.subheader("Your Answer")
            user_answer = st.text_area("Write your answer here:", height=200)
            
            if st.button("Submit Answer"):
                if user_answer.strip():
                    with chat_container:
                        message(user_answer, is_user=True, key=f"answer_{len(st.session_state.messages)}")
                    
                    add_message("user", user_answer)
                    st.session_state.hist_for_context.append(HumanMessage(content=user_answer))
                    
                    with st.spinner("Grading your answer..."):
                        grade_prompt = f"Here is the question: {st.session_state.current_question}.\nHere is student's answer: {user_answer}"
                        feedback = grade_agent.generate_answer(query=grade_prompt, mode="normal")
                    
                    with chat_container:
                        message(feedback, is_user=False, key=f"feedback_{len(st.session_state.messages)}", avatar_style='bottts-neutral')
                    
                    add_message("assistant", feedback)
                    st.session_state.hist_for_context.append(AIMessage(content=feedback))
                    st.session_state.answer_submitted = True
                    st.rerun()
                else:
                    st.error("Please write an answer before submitting.")
    
    # Chat input for follow-up questions
    if st.session_state.answer_submitted:
        prompt = st.chat_input("Ask a follow-up question or request clarification", disabled=st.session_state.disable_input)
        
        if prompt is not None and prompt.strip() != "":
            with chat_container:
                message(prompt, is_user=True, key=f"followup_{len(st.session_state.messages)}")
            
            add_message("user", prompt)
            st.session_state.hist_for_context.append(HumanMessage(content=prompt))
            
            with st.spinner("Thinking..."):
                query_aware_history = ha.get_hist_context(st.session_state.hist_for_context, prompt)
                rel_docs = r.get_context(query_aware_history)
                
                response = question_agent.generate_answer(prompt, rel_docs)
            
            with chat_container:
                message(response, is_user=False, key=f"response_{len(st.session_state.messages)}", avatar_style='bottts-neutral')
            
            add_message("assistant", response)
            st.session_state.hist_for_context.append(AIMessage(content=response))

if __name__ == "__main__":
    main()