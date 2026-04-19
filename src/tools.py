"""Tool calling utilities for RAG Agent."""

from .vectorstore import VectorStore, SearchResult


def create_rag_tool(vectorstore: VectorStore, top_k: int = 5):
    """Create a RAG search tool for use with LLM agents.

    Args:
        vectorstore: The vector store to query
        top_k: Number of results to return (from config)

    Returns:
        A dict representing the tool schema for OpenAI
    """
    def search_knowledge_base(query: str) -> str:
        """
        从向量知识库中检索与查询相关的文档片段。

        适用于需要引用具体文档内容来回答的问题。
        返回最相关的文档片段及其来源信息。
        """
        results = vectorstore.search(query, top_k=top_k)

        if not results:
            return "未找到相关文档"

        formatted = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            formatted.append(
                f"[文档{i}] (相似度距离: {r.distance:.3f})\n"
                f"内容: {r.content}\n"
                f"来源: {source}"
            )

        return "\n\n".join(formatted)

    # Attach the function to return as a callable with metadata
    search_knowledge_base.tool_name = "search_knowledge_base"
    search_knowledge_base.tool_description = "从向量知识库中检索与查询相关的文档片段。适用于需要引用具体文档内容来回答的问题。返回最相关的文档片段及其来源信息。"

    return search_knowledge_base


def get_tool_schemas():
    """Return the OpenAI tool schemas for RAG tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "从向量知识库中检索与查询相关的文档片段。适用于需要引用具体文档内容来回答的问题。返回最相关的文档片段及其来源信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "自然语言查询，用于检索知识库",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]
