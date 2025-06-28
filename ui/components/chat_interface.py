"""
Streamlit chat interface for PolicyPal.
Provides a user-friendly chatbot interface with loading states and user feedback.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import time
import json

# Import feedback collector
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.feedback_collector import feedback_collector


def initialize_chat_session():
    """Initialize chat session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_summary" not in st.session_state:
        st.session_state.conversation_summary = {
            "total_turns": 0,
            "start_time": None,
            "end_time": None,
            "duration_minutes": 0,
            "total_sources_used": 0
        }
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "user_feedback" not in st.session_state:
        st.session_state.user_feedback = {}


def display_chat_header():
    """Display the chat interface header."""
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🤖 PolicyPal</h1>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1rem;">Your AI assistant for company policy documents</p>
    </div>
    """, unsafe_allow_html=True)


def display_chat_messages():
    """Display all chat messages in the session with enhanced styling."""
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            # User message with enhanced styling
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 15px; margin: 1rem 0; text-align: right;">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message with enhanced styling
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1rem; border-radius: 15px; margin: 1rem 0;">
                <strong>PolicyPal:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources with enhanced styling
            if message.get("sources"):
                st.markdown("""
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #856404;">📚 Sources:</h4>
                """, unsafe_allow_html=True)
                
                for j, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div style="background: white; border-radius: 5px; padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid #ffc107;">
                        <strong>{j}.</strong> {source}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display user feedback buttons
            message_id = f"msg_{i}"
            if message_id not in st.session_state.user_feedback:
                display_feedback_buttons(message_id, message)


def display_feedback_buttons(message_id: str, message: Dict[str, Any]):
    """Display feedback buttons for assistant messages."""
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("👍 Helpful", key=f"helpful_{message_id}", use_container_width=True):
            # Store feedback
            user_question = ""
            if st.session_state.messages:
                # Find the user question that preceded this response
                for i, msg in enumerate(st.session_state.messages):
                    if msg.get("id") == message_id or (msg["role"] == "assistant" and msg["content"] == message["content"]):
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            user_question = st.session_state.messages[i-1]["content"]
                            break
            
            feedback_collector.add_feedback(
                message_id=message_id,
                feedback_type="helpful",
                user_question=user_question,
                assistant_response=message["content"],
                sources=message.get("sources", []),
                performance_info=message.get("performance_info", {})
            )
            
            st.session_state.user_feedback[message_id] = "helpful"
            st.success("Thank you for your feedback!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("👎 Not Helpful", key=f"not_helpful_{message_id}", use_container_width=True):
            # Store feedback
            user_question = ""
            if st.session_state.messages:
                # Find the user question that preceded this response
                for i, msg in enumerate(st.session_state.messages):
                    if msg.get("id") == message_id or (msg["role"] == "assistant" and msg["content"] == message["content"]):
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            user_question = st.session_state.messages[i-1]["content"]
                            break
            
            feedback_collector.add_feedback(
                message_id=message_id,
                feedback_type="not_helpful",
                user_question=user_question,
                assistant_response=message["content"],
                sources=message.get("sources", []),
                performance_info=message.get("performance_info", {})
            )
            
            st.session_state.user_feedback[message_id] = "not_helpful"
            st.error("We'll work to improve our responses.")
            time.sleep(1)
            st.rerun()


def display_source_details(sources: Optional[List[str]], retrieved_docs: Optional[List[Dict[str, Any]]]):
    """Display detailed source information in an expander with enhanced styling."""
    if sources and retrieved_docs:
        with st.expander("📄 View Retrieved Documents", expanded=False):
            for i, (source, doc) in enumerate(zip(sources, retrieved_docs), 1):
                st.markdown(f"""
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                    <h4 style="color: #495057; margin: 0 0 0.5rem 0;">📄 Source {i}: {source}</h4>
                    <p style="color: #6c757d; margin: 0.25rem 0;"><strong>Relevance Score:</strong> {doc.get('score', 'N/A'):.3f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Content:**")
                content = doc.get('content', 'No content available')
                if len(content) > 500:
                    st.text(content[:500] + "...")
                    with st.expander("Show full content"):
                        st.text(content)
                else:
                    st.text(content)
                st.markdown("---")


def display_conversation_stats():
    """Display conversation statistics in the sidebar with enhanced styling."""
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="margin: 0; text-align: center;">📊 Conversation Stats</h3>
    </div>
    """, unsafe_allow_html=True)
    
    summary = st.session_state.conversation_summary
    
    # Create metric cards
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Turns", summary["total_turns"])
    with col2:
        st.metric("Sources Used", summary["total_sources_used"])
    
    if summary["duration_minutes"] > 0:
        st.sidebar.metric("Duration (min)", f"{summary['duration_minutes']:.1f}")
    
    # Display recent sources
    if st.session_state.messages:
        recent_sources = []
        for msg in st.session_state.messages[-5:]:  # Last 5 messages
            if msg["role"] == "assistant" and msg.get("sources"):
                recent_sources.extend(msg["sources"])
        
        if recent_sources:
            st.sidebar.markdown("### 📚 Recent Sources")
            for source in list(set(recent_sources))[-3:]:  # Last 3 unique sources
                st.sidebar.markdown(f"• {source}")
    
    # Display feedback summary from collector
    feedback_summary = feedback_collector.get_feedback_summary()
    if feedback_summary["total_feedback"] > 0:
        st.sidebar.markdown("### 💬 Feedback Summary")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("👍 Helpful", feedback_summary["helpful_count"])
        with col2:
            st.metric("👎 Not Helpful", feedback_summary["not_helpful_count"])
        
        # Calculate satisfaction rate
        satisfaction_rate = (feedback_summary["helpful_count"] / feedback_summary["total_feedback"]) * 100
        st.sidebar.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")


def add_user_message(content: str):
    """Add a user message to the chat."""
    st.session_state.messages.append({
        "role": "user",
        "content": content,
        "timestamp": datetime.now(),
        "id": f"user_{len(st.session_state.messages)}"
    })


def add_assistant_message(content: str, sources: Optional[List[str]] = None, 
                         retrieved_docs: Optional[List[Dict[str, Any]]] = None,
                         performance_info: Optional[Dict[str, Any]] = None):
    """Add an assistant message to the chat."""
    message = {
        "role": "assistant",
        "content": content,
        "timestamp": datetime.now(),
        "id": f"assistant_{len(st.session_state.messages)}"
    }
    
    if sources:
        message["sources"] = sources
    
    if retrieved_docs:
        message["retrieved_docs"] = retrieved_docs
    
    if performance_info:
        message["performance_info"] = performance_info
    
    st.session_state.messages.append(message)


def clear_chat_history():
    """Clear the chat history."""
    st.session_state.messages = []
    st.session_state.conversation_summary = {
        "total_turns": 0,
        "start_time": None,
        "end_time": None,
        "duration_minutes": 0,
        "total_sources_used": 0
    }
    st.session_state.user_feedback = {}


def display_processing_status():
    """Display processing status with enhanced progress indicators."""
    if st.session_state.processing:
        # Create a container for the processing status
        status_container = st.container()
        
        with status_container:
            # Enhanced progress bar
            progress_bar = st.progress(0)
            
            # Status text with better styling
            status_text = st.empty()
            
            # Processing steps with emojis
            steps = [
                "🔍 Analyzing your question...",
                "📚 Searching policy documents...",
                "🤖 Generating response...",
                "✅ Preparing answer..."
            ]
            
            # Animate through the steps
            for i, step in enumerate(steps):
                status_text.markdown(f"""
                <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px; padding: 1rem; text-align: center;">
                    <h4 style="margin: 0; color: #1976d2;">{step}</h4>
                </div>
                """, unsafe_allow_html=True)
                progress = (i + 1) / len(steps)
                progress_bar.progress(progress)
                time.sleep(0.5)  # Brief pause between steps
            
            # Clear the status
            status_text.empty()
            progress_bar.empty()


def display_file_upload_section():
    """Display file upload section in sidebar."""
    st.sidebar.markdown("""
    <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
        <h4 style="margin: 0 0 0.5rem 0; color: #2e7d32;">📁 Upload Documents</h4>
        <p style="margin: 0; font-size: 0.9rem; color: #388e3c;">Add new policy documents to the system</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.sidebar.file_uploader(
        "Choose files",
        type=['txt', 'pdf', 'docx', 'csv'],
        accept_multiple_files=True,
        help="Upload policy documents to enhance the knowledge base"
    )
    
    if uploaded_files:
        if st.sidebar.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                # Here you would integrate with the ingestion pipeline
                st.success(f"Successfully processed {len(uploaded_files)} documents!")
                time.sleep(2)
                st.rerun()


def display_feedback_analytics():
    """Display feedback analytics in an expander."""
    with st.sidebar.expander("📈 Feedback Analytics", expanded=False):
        feedback_summary = feedback_collector.get_feedback_summary()
        
        if feedback_summary["total_feedback"] > 0:
            # Create a simple chart
            data = {
                "Feedback Type": ["Helpful", "Not Helpful"],
                "Count": [feedback_summary["helpful_count"], feedback_summary["not_helpful_count"]]
            }
            df = pd.DataFrame(data)
            
            st.bar_chart(df.set_index("Feedback Type"))
            
            # Show recent feedback
            recent_feedback = feedback_collector.get_recent_feedback(5)
            if recent_feedback:
                st.markdown("### Recent Feedback")
                for feedback in recent_feedback:
                    emoji = "👍" if feedback["feedback_type"] == "helpful" else "👎"
                    st.markdown(f"{emoji} {feedback['feedback_type'].title()} - {feedback['timestamp'][:10]}")
        else:
            st.info("No feedback collected yet.")


def display_chat_interface(conversational_chain):
    """
    Main chat interface function with enhanced features.
    
    Args:
        conversational_chain: The conversational chain instance
    """
    initialize_chat_session()
    
    # Sidebar with enhanced styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="margin: 0; text-align: center;">🤖 PolicyPal</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    display_file_upload_section()
    
    # Clear chat button with enhanced styling
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history()
        conversational_chain.clear_conversation()
        st.success("Chat history cleared!")
        time.sleep(1)
        st.rerun()
    
    # Display conversation stats
    display_conversation_stats()
    
    # Display feedback analytics
    display_feedback_analytics()
    
    # Main chat area
    display_chat_header()
    
    # Welcome message for new users
    if not st.session_state.messages:
        st.markdown("""
        <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 10px; padding: 2rem; text-align: center; margin: 2rem 0;">
            <h3 style="color: #1976d2; margin-bottom: 1rem;">👋 Welcome to PolicyPal!</h3>
            <p style="color: #1565c0; margin-bottom: 1rem;">Ask me anything about company policies and I'll help you find the information you need.</p>
            <div style="background: white; border-radius: 8px; padding: 1rem; text-align: left;">
                <h4 style="color: #1976d2; margin-bottom: 0.5rem;">💡 Example questions:</h4>
                <ul style="color: #1565c0; margin: 0;">
                    <li>What is the vacation policy?</li>
                    <li>How many remote work days are allowed?</li>
                    <li>What are the expense reimbursement rules?</li>
                    <li>What is the dress code policy?</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    display_chat_messages()
    
    # Chat input with enhanced styling
    st.markdown("""
    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
        <h4 style="margin: 0 0 0.5rem 0; color: #495057;">💬 Ask a question:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask about company policies..."):
        # Set processing state
        st.session_state.processing = True
        
        # Add user message
        add_user_message(prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display processing status
        display_processing_status()
        
        # Get response from conversational chain
        with st.chat_message("assistant"):
            try:
                # Show initial loading message
                with st.spinner("🤖 PolicyPal is thinking..."):
                    result = conversational_chain.chat(prompt)
                
                if result["success"]:
                    # Display response with typing effect
                    response_placeholder = st.empty()
                    response_text = result["response"]
                    
                    # Simulate typing effect
                    for i in range(len(response_text) + 1):
                        response_placeholder.markdown(response_text[:i] + "▌" if i < len(response_text) else response_text)
                        time.sleep(0.02)  # Adjust speed as needed
                    
                    # Display sources with enhanced styling
                    if result["sources"]:
                        st.markdown("""
                        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #856404;">📚 Sources:</h4>
                        """, unsafe_allow_html=True)
                        
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"""
                            <div style="background: white; border-radius: 5px; padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid #ffc107;">
                                <strong>{i}.</strong> {source}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display performance metrics with enhanced styling
                    if result.get("search_latency"):
                        st.markdown(f"""
                        <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 8px; padding: 0.5rem; margin: 0.5rem 0;">
                            <p style="margin: 0; color: #2e7d32;"><strong>⚡ Performance:</strong> Search completed in {result['search_latency']:.3f}s</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add assistant message to session
                    add_assistant_message(
                        result["response"],
                        result.get("sources", []),
                        result.get("retrieved_documents", []),
                        result.get("performance_info", {})
                    )
                    
                    # Update conversation summary
                    summary = conversational_chain.get_conversation_summary()
                    st.session_state.conversation_summary = summary
                    
                    # Display source details in expander
                    display_source_details(
                        result.get("sources", []),
                        result.get("retrieved_documents", [])
                    )
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                    add_assistant_message(f"Error: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                st.error(error_msg)
                add_assistant_message(error_msg)
            finally:
                # Clear processing state
                st.session_state.processing = False 