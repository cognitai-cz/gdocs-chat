from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool, InjectedToolArg
from sqlalchemy.sql.annotation import Annotated


@tool
def rules_context_tool(query: str, config_param: RunnableConfig) -> str:
    """Get rules context from the CSST official rules"""
    vector_store: Chroma = config_param["configurable"]["vector_store"]
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])
