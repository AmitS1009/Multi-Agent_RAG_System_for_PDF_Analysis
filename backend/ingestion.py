import pdfplumber
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdf(file_path: str) -> List[Document]:
    """Loads a PDF and returns a list of LangChain Documents (one per page)."""
    documents = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "page": i + 1}
                    ))
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)
