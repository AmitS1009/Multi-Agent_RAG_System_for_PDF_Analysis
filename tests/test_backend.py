import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ingestion import load_pdf, split_documents
from backend.vector_store import add_documents_to_store, query_vector_store
from agents.planner import PlannerAgent

def test_backend():
    print("Testing Backend...")
    # Mock PDF loading (since we might not have a PDF handy, we'll create a dummy doc)
    from langchain_core.documents import Document
    docs = [
        Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "test.pdf", "page": 1}),
        Document(page_content="Artificial Intelligence is transforming the world.", metadata={"source": "test.pdf", "page": 2})
    ]
    
    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Chunks created: {len(chunks)}")
    
    print("Adding to vector store...")
    try:
        add_documents_to_store(chunks)
        print("Added to vector store.")
    except Exception as e:
        print(f"Vector store error (expected if no API key): {e}")
        return

    print("Querying vector store...")
    results = query_vector_store("fox")
    print(f"Query results: {len(results)}")
    
    print("Testing Planner...")
    try:
        planner = PlannerAgent()
        # Mock input
        input_data = {"query": "Summarize the documents", "documents": docs}
        # We won't actually run process because it calls LLM and costs money/needs key
        # But we can check if it initialized
        print("Planner initialized successfully.")
    except Exception as e:
        print(f"Planner initialization error (expected if no API key): {e}")

if __name__ == "__main__":
    test_backend()
