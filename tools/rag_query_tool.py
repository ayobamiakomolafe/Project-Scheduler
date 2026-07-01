import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from crewai.tools import tool

load_dotenv()


def make_rag_tool(vector_store: Chroma):
    """
    Wraps a given Chroma vector store into a CrewAI tool that the
    Estimator agent can call to look up real company salary/role data.

    Returns None if vector_store is None, so callers can simply do:
        tools=[rag_tool] if rag_tool else []
    """
    if vector_store is None:
        return None

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    @tool("Query Pre-existing Vector DB")
    def query_existing_db(query: str) -> str:
        """
        Searches the company employee/salary database for relevant documents
        matching the given search query. Use this whenever you need real
        salary rates or role/seniority information.
        """
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return query_existing_db


def load_persisted_rag_tool(persist_directory: str = "chroma_db", openai_api_key: str = None):
    """
    Convenience helper for loading a previously-persisted Chroma DB from disk
    (e.g. a default company_employee_data.md DB that ships with the app),
    and wrapping it as a tool. Returns None if the directory doesn't exist.
    """
    if not os.path.isdir(persist_directory):
        return None

    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key or os.getenv("OPENAI_KEY")
    )
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    return make_rag_tool(vector_store)
