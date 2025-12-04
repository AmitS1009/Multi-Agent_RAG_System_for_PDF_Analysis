import os
from typing import Any, Dict
from .base import BaseAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

class SummarizationAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Summarization Agent")
        if os.getenv("GOOGLE_API_KEY"):
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        elif os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model="gpt-4o")
        else:
            raise ValueError("No API key found for LLM.")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        documents = input_data.get("documents")
        if not documents:
            return {"error": "No documents provided for summarization"}

        # Use map-reduce for large documents, or stuff for small ones.
        # For simplicity, we'll use map-reduce if total length is large, else stuff.
        # Here we just use map_reduce for robustness.
        
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        summary = chain.invoke(documents)
        
        return {
            "response": summary["output_text"],
            "evidence": [] # Summaries generally reference the whole doc
        }
