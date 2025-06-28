"""
Streamlit chat interface for PolicyPal.
Provides a user-friendly chatbot interface with loading states.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import time


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


def display_chat_header():
    """Display the chat interface header."""
    st.title("🤖 PolicyPal")
    st.markdown("**Your AI assistant for company policy documents**")
    st.markdown("---")


def display_chat_messages():
    """Display all chat messages in the session."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and message.get("sources"):
                st.markdown("**Sources:**")
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"{i}. {source}")


def display_source_details(sources: Optional[List[str]], retrieved_docs: Optional[List[Dict[str, Any]]]):
    """Display detailed source information in an expander."""
    if sources and retrieved_docs:
        with st.expander("📄 View Retrieved Documents"):
            for i, (source, doc) in enumerate(zip(sources, retrieved_docs), 1):
                st.markdown(f"**Source {i}: {source}**")
                st.markdown(f"**Score:** {doc.get('score', 'N/A'):.3f}")
                st.markdown(f"**Content:**")
                st.text(doc.get('content', 'No content available')[:500] + "..." if len(doc.get('content', '')) > 500 else doc.get('content', 'No content available'))
                st.markdown("---")


def display_conversation_stats():
    """Display conversation statistics in the sidebar."""
    st.sidebar.markdown("## 📊 Conversation Stats")
    
    summary = st.session_state.conversation_summary
    st.sidebar.metric("Total Turns", summary["total_turns"])
    st.sidebar.metric("Sources Used", summary["total_sources_used"])
    
    if summary["duration_minutes"] > 0:
        st.sidebar.metric("Duration (min)", f"{summary['duration_minutes']:.1f}")
    
    # Display recent sources
    if st.session_state.messages:
        recent_sources = []
        for msg in st.session_state.messages[-5:]:  # Last 5 messages
            if msg["role"] == "assistant" and msg.get("sources"):
                recent_sources.extend(msg["sources"])
        
        if recent_sources:
            st.sidebar.markdown("### Recent Sources")
            for source in list(set(recent_sources))[-3:]:  # Last 3 unique sources
                st.sidebar.markdown(f"• {source}")


def add_user_message(content: str):
    """Add a user message to the chat."""
    st.session_state.messages.append({
        "role": "user",
        "content": content,
        "timestamp": datetime.now()
    })


def add_assistant_message(content: str, sources: Optional[List[str]] = None, 
                         retrieved_docs: Optional[List[Dict[str, Any]]] = None):
    """Add an assistant message to the chat."""
    message = {
        "role": "assistant",
        "content": content,
        "timestamp": datetime.now()
    }
    
    if sources:
        message["sources"] = sources
    
    if retrieved_docs:
        message["retrieved_docs"] = retrieved_docs
    
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


def display_processing_status():
    """Display processing status with progress indicators."""
    if st.session_state.processing:
        # Create a container for the processing status
        status_container = st.container()
        
        with status_container:
            # Progress bar
            progress_bar = st.progress(0)
            
            # Status text
            status_text = st.empty()
            
            # Processing steps
            steps = [
                "🔍 Analyzing your question...",
                "📚 Searching policy documents...",
                "🤖 Generating response...",
                "✅ Preparing answer..."
            ]
            
            # Animate through the steps
            for i, step in enumerate(steps):
                status_text.text(step)
                progress = (i + 1) / len(steps)
                progress_bar.progress(progress)
                time.sleep(0.5)  # Brief pause between steps
            
            # Clear the status
            status_text.empty()
            progress_bar.empty()


def display_chat_interface(conversational_chain):
    """
    Main chat interface function.
    
    Args:
        conversational_chain: The conversational chain instance
    """
    initialize_chat_session()
    
    # Sidebar
    st.sidebar.title("PolicyPal")
    st.sidebar.markdown("---")
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        clear_chat_history()
        conversational_chain.clear_conversation()
        st.rerun()
    
    # Display conversation stats
    display_conversation_stats()
    
    # Main chat area
    display_chat_header()
    display_chat_messages()
    
    # Chat input
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
                    
                    # Display sources
                    if result["sources"]:
                        st.markdown("**📚 Sources:**")
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"{i}. {source}")
                    
                    # Display performance metrics
                    if result.get("search_latency"):
                        st.markdown(f"**⚡ Performance:** Search completed in {result['search_latency']:.3f}s")
                    
                    # Add assistant message to session
                    add_assistant_message(
                        result["response"],
                        result.get("sources", []),
                        result.get("retrieved_documents", [])
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
    
    # Remove auto-rerun to prevent UI issues
    # if st.session_state.messages:
    #     st.rerun() 