import os
from typing import Any, Dict, List
from .base import BaseAgent
from .rag_agent import RAGAgent
from .summarization_agent import SummarizationAgent
from .reasoning_agents import ComparatorAgent, TimelineAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Plan(BaseModel):
    intent: str = Field(description="The user's intent: 'query', 'summarize', 'compare', 'timeline'")
    reasoning: str = Field(description="Explanation of why this plan was chosen")

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Planner Agent")
        if os.getenv("GOOGLE_API_KEY"):
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        elif os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model="gpt-4o")
        else:
            raise ValueError("No API key found for LLM.")
        
        self.rag_agent = RAGAgent()
        self.summarization_agent = SummarizationAgent()
        self.comparator_agent = ComparatorAgent()
        self.timeline_agent = TimelineAgent()

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query")
        documents = input_data.get("documents") # For summarization

        # 1. Detect Intent
        parser = JsonOutputParser(pydantic_object=Plan)
        template = """Determine the user's intent based on the query.
        Options:
        - 'query': General question answering about the documents.
        - 'summarize': Request for a summary of the documents.
        - 'compare': Request to compare items or sections.
        - 'timeline': Request for a timeline of events.

        Query: {query}

        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_template(template, partial_variables={"format_instructions": parser.get_format_instructions()})
        chain = prompt | self.llm | parser
        
        try:
            plan = chain.invoke({"query": query})
            intent = plan["intent"]
        except Exception as e:
            print(f"Planning error: {e}")
            intent = "query" # Fallback

        # 2. Route to Agents
        response = {}
        if intent == "summarize":
            response = self.summarization_agent.process({"documents": documents})
            response["trace"] = "Planner -> Summarization Agent"
        
        elif intent == "compare":
            # Chain: RAG -> Comparator
            rag_result = self.rag_agent.process({"query": query})
            context = rag_result.get("response") # Use the RAG answer as context for comparison
            # Ideally we pass the retrieved chunks, but RAG answer is a good synthesized start.
            # Better: Pass the raw retrieved chunks.
            evidence_text = "\n".join([e["content"] for e in rag_result.get("evidence", [])])
            
            comp_result = self.comparator_agent.process({"context": evidence_text, "query": query})
            response = {
                "response": comp_result["response"],
                "evidence": rag_result.get("evidence"),
                "trace": "Planner -> RAG Agent -> Comparator Agent"
            }

        elif intent == "timeline":
             # Chain: RAG -> Timeline
            rag_result = self.rag_agent.process({"query": query})
            evidence_text = "\n".join([e["content"] for e in rag_result.get("evidence", [])])
            
            timeline_result = self.timeline_agent.process({"context": evidence_text, "query": query})
            response = {
                "response": timeline_result["response"],
                "evidence": rag_result.get("evidence"),
                "trace": "Planner -> RAG Agent -> Timeline Agent"
            }

        else: # Default to query/RAG
            response = self.rag_agent.process({"query": query})
            response["trace"] = "Planner -> RAG Agent"

        return response
