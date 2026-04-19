
import os
import math
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
import uuid

# =========================================================
# All expensive initialisations inside @st.cache_resource
# =========================================================
@st.cache_resource
def initialise_agent():
    """Build KB, graph, and return compiled app. Called once per session."""
    from agent import build_agent
    app, embedder, collection = build_agent()
    return app, embedder, collection

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Physics Study Buddy",
    page_icon="⚛️",
    layout="wide"
)

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.title("⚛️ Physics Study Buddy")
    st.markdown("**Agentic AI Capstone 2026**")
    st.markdown("---")
    st.markdown("**Domain:** B.Tech Physics")
    st.markdown("**Topics covered:**")
    topics = [
        "Newton's Laws of Motion",
        "Work, Energy and Power",
        "Laws of Thermodynamics",
        "Electric Current & Ohm's Law",
        "Capacitors and Capacitance",
        "Magnetic Force & Faraday's Law",
        "Wave Motion and Sound",
        "Optics — Reflection & Refraction",
        "Modern Physics & Photoelectric Effect",
        "Gravitation and Kepler's Laws",
    ]
    for t in topics:
        st.markdown(f"• {t}")
    st.markdown("---")
    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.markdown("---")
    st.caption("Dr. Kanthi Kiran Sirra | Agentic AI Course 2026")

# =========================================================
# Session state init
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# =========================================================
# Main chat UI
# =========================================================
st.title("⚛️ Physics Study Buddy")
st.caption("Ask me anything from your B.Tech Physics syllabus!")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Load agent
app, embedder, collection = initialise_agent()

# Chat input
if prompt := st.chat_input("Ask a physics question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            initial_state = {
                "question": prompt,
                "messages": st.session_state.messages,  # pass history for memory
                "route": "",
                "retrieved": "",
                "sources": [],
                "tool_result": "",
                "answer": "",
                "faithfulness": 0.0,
                "eval_retries": 0,
                "user_name": ""
            }
            result = app.invoke(initial_state, config=config)
            answer = result.get("answer", "I could not generate a response. Please try again.")
            sources = result.get("sources", [])
            faith   = result.get("faithfulness", 0.0)

        st.markdown(answer)
        if sources:
            st.caption(f"📚 Sources: {', '.join(sources)} | Faithfulness: {faith:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
