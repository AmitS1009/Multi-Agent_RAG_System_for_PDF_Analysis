import os
from typing import List
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Global vector store instance
vector_store = None

def get_embeddings_model():
    """Returns the embeddings model based on environment variables."""
    if os.getenv("GOOGLE_API_KEY"):
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings()
    else:
        raise ValueError("No API key found for embeddings. Set GOOGLE_API_KEY or OPENAI_API_KEY.")

def initialize_vector_store(persist_directory: str = "./chroma_db"):
    """Initializes the Chroma vector store."""
    global vector_store
    embeddings = get_embeddings_model()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="pdf_rag"
    )
    return vector_store

def add_documents_to_store(documents: List[Document]):
    """Adds documents to the vector store."""
    global vector_store
    if vector_store is None:
        initialize_vector_store()
    vector_store.add_documents(documents)

def query_vector_store(query: str, k: int = 5) -> List[Document]:
    """Queries the vector store for similar documents."""
    global vector_store
    if vector_store is None:
        initialize_vector_store()
    return vector_store.similarity_search(query, k=k)
