import os
from typing import Any, Dict
from .base import BaseAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ComparatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Comparator Agent")
        if os.getenv("GOOGLE_API_KEY"):
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        elif os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model="gpt-4o")
        else:
            raise ValueError("No API key found for LLM.")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        context = input_data.get("context") # Expecting context from RAG or other source
        query = input_data.get("query")

        template = """Compare and contrast the information based on the following context:

        {context}

        Request: {query}

        Highlight similarities and differences.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"context": context, "query": query})
        return {"response": response}

class TimelineAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Timeline Agent")
        if os.getenv("GOOGLE_API_KEY"):
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        elif os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model="gpt-4o")
        else:
            raise ValueError("No API key found for LLM.")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        context = input_data.get("context")
        query = input_data.get("query")

        template = """Create a chronological timeline of events based on the following context:

        {context}

        Request: {query}

        Format as a list of events with dates/times if available.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"context": context, "query": query})
        return {"response": response}
