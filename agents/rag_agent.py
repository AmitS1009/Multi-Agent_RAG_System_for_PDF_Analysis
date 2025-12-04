import os
from typing import Any, Dict
from .base import BaseAgent
from backend.vector_store import query_vector_store
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="RAG Agent")
        if os.getenv("GOOGLE_API_KEY"):
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        elif os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model="gpt-4o")
        else:
            raise ValueError("No API key found for LLM.")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query")
        if not query:
            return {"error": "No query provided"}

        # Retrieve documents
        docs = query_vector_store(query)
        
        # Format context
        context = "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', '?')})\nContent: {d.page_content}" for d in docs])

        # Generate answer
        template = """Answer the question based only on the following context:

        {context}

        Question: {question}

        Provide a detailed answer and cite the sources (filename and page number) for your information.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"context": context, "question": query})

        return {
            "response": response,
            "evidence": [{"source": d.metadata.get("source"), "page": d.metadata.get("page"), "content": d.page_content} for d in docs]
        }
