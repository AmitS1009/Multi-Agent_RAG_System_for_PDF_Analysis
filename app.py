import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from backend.ingestion import load_pdf, split_documents
from backend.vector_store import add_documents_to_store
from agents.planner import PlannerAgent

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Multi-Agent PDF Analysis", layout="wide")

# Initialize Planner
if "planner" not in st.session_state:
    try:
        st.session_state.planner = PlannerAgent()
    except Exception as e:
        st.error(f"Failed to initialize agents: {e}. Please check your API keys.")

st.title("Multi-Agent PDF Analysis System")

# Sidebar for Upload and Navigation
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                all_docs = []
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load and split
                    docs = load_pdf(tmp_path)
                    # Update metadata with original filename
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    
                    all_docs.extend(docs)
                    os.remove(tmp_path) # Clean up temp file
                
                # Split and Store
                chunks = split_documents(all_docs)
                add_documents_to_store(chunks)
                
                # Store full docs in session state for summarization/viewer
                st.session_state.documents = all_docs
                st.success(f"Processed {len(uploaded_files)} files ({len(chunks)} chunks).")

    st.header("Document Navigator")
    if "documents" in st.session_state and st.session_state.documents:
        selected_doc = st.selectbox("Select Document", list(set(d.metadata["source"] for d in st.session_state.documents)))
        
        # Filter docs for selected file
        file_docs = [d for d in st.session_state.documents if d.metadata["source"] == selected_doc]
        
        # Page selector
        page_num = st.number_input("Page", min_value=1, max_value=len(file_docs), value=1)
        
        # Display content
        if file_docs:
            st.text_area("Page Content", file_docs[page_num-1].page_content, height=400)

# Chat Interface
st.chat_message("assistant").write("Hello! Upload some PDFs and ask me anything about them.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "evidence" in message:
            with st.expander("View Evidence"):
                for i, ev in enumerate(message["evidence"]):
                    st.markdown(f"**{i+1}. {ev.get('source')} (Page {ev.get('page')})**")
                    st.text(ev.get("content")[:200] + "...")

if prompt := st.chat_input("Ask a question or request a summary..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if "planner" in st.session_state:
                # Prepare input
                input_data = {
                    "query": prompt,
                    "documents": st.session_state.get("documents", [])
                }
                
                result = st.session_state.planner.process(input_data)
                
                response_text = result.get("response", "I couldn't generate a response.")
                evidence = result.get("evidence", [])
                trace = result.get("trace", "Unknown")
                
                st.markdown(response_text)
                st.caption(f"Trace: {trace}")
                
                if evidence:
                    with st.expander("View Evidence"):
                        for i, ev in enumerate(evidence):
                            st.markdown(f"**{i+1}. {ev.get('source')} (Page {ev.get('page')})**")
                            st.text(ev.get("content")[:200] + "...")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "evidence": evidence
                })
            else:
                st.error("Agent system not initialized.")
