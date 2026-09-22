import streamlit as st
import os
from backend import EnterpriseRAGBackend

# Streamlit Page Setup
st.set_page_config(page_title="ChromaQuery Enterprise AI", layout="wide", page_icon="⚡")

st.title("⚡ ChromaQuery – Enterprise Knowledge Platform")
st.caption("Grounding LLM responses using web scraping, vector embeddings, and ChromaDB.")

# Initialize Session State
if "rag" not in st.session_state:
    st.session_state.rag = EnterpriseRAGBackend()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Knowledge Ingestion Manager
with st.sidebar:
    st.header("📥 Ingestion Control Center")
    st.markdown("Add unstructured data to your persistent ChromaDB collection.")
    
    tab1, tab2 = st.tabs(["Web URL", "Text Document"])
    
    with tab1:
        url_input = st.text_input("Enter Web URL:")
        if st.button("Scrape & Index Web Page"):
            if url_input:
                with st.spinner("Scraping and vectorizing content..."):
                    try:
                        chunks_count = st.session_state.rag.ingest_url(url_input)
                        st.success(f"Successfully indexed {chunks_count} chunks into ChromaDB!")
                    except Exception as e:
                        st.error(f"Failed to scrape: {e}")
            else:
                st.warning("Please enter a valid URL.")

    with tab2:
        doc_title = st.text_input("Document Name/Title:")
        doc_text = st.text_area("Paste Content:")
        if st.button("Index Text Document"):
            if doc_title and doc_text:
                with st.spinner("Vectorizing text document..."):
                    chunks_count = st.session_state.rag.ingest_raw_text(doc_text, doc_title)
                    st.success(f"Indexed {chunks_count} chunks from '{doc_title}'!")
            else:
                st.warning("Provide both a title and text body.")

# Main Interface - Interactive Chat
st.subheader("💬 Query Enterprise Knowledge Base")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Verified Context Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

# Handle User Input
if user_prompt := st.chat_input("Ask a question based on indexed knowledge..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # Process via RAG Backend
    with st.chat_message("assistant"):
        with st.spinner("Searching vectors & generating answer..."):
            result = st.session_state.rag.answer_query(user_prompt)
            st.write(result["answer"])
            
            if result["sources"]:
                with st.expander("🔍 Verified Context Sources"):
                    for source in result["sources"]:
                        st.markdown(f"- `{source}`")

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })
