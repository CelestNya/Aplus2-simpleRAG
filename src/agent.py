"""LLM Agent with Tool Calling support."""

import json

from openai import OpenAI

from .config import Config
from .tools import create_rag_tool, get_tool_schemas
from .vectorstore import VectorStore


class RAGAgent:
    """RAG Agent that uses Tool Calling to search knowledge base."""

    SYSTEM_PROMPT = """你是一个有用的RAG助手，可以访问知识库。
当用户提问时，使用search_knowledge_base工具来查找相关信息后再回答。
请始终引用你的信息来源。"""

    def __init__(self, config: Config, vectorstore: VectorStore):
        """Initialize the RAG agent.

        Args:
            config: Configuration object with LLM settings
            vectorstore: Vector store to search
        """
        self.config = config
        self.vectorstore = vectorstore

        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
        )
        self.model = config.llm.model

        # Create RAG tool
        self.rag_tool = create_rag_tool(
            vectorstore,
            top_k=config.search.default_top_k,
        )
        self.tools = get_tool_schemas()

    def chat(self, query: str, conversation_history: list[dict] = None) -> str:
        """Chat with the agent.

        Args:
            query: User query
            conversation_history: Optional conversation history

        Returns:
            Agent's response
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})

        # First call - let LLM decide if it needs to call tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message

        # Handle tool calls
        if assistant_msg.tool_calls:
            messages.append(assistant_msg.model_dump())

            # Execute tools and add results
            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if tool_name == "search_knowledge_base":
                    result = self.rag_tool(**tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Second call - generate final response with tool results
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content

        return assistant_msg.content
