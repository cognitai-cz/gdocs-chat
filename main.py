from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings

from src.config.settings import app_settings
from src.embeddings.pdf_embedding import PDFEmbedding
from src.logging import setup_logging
from src.rag.graph import Rag


def main():
    embeddings = OpenAIEmbeddings(model=app_settings.EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=app_settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(app_settings.CHROMA_DB_FILE_PATH),
    )

    setup_logging()
    pdf_embed = PDFEmbedding(vector_store)
    pdf_embed.load_to_db()

    model = init_chat_model(
        app_settings.OPENAI_MODEL, model_provider=app_settings.LANGSMITH_MODEL_PROVIDER
    )

    config = {"configurable": {"thread_id": "123", "llm": model}}

    rag = Rag.build(vector_store)
    while True:
        res = rag.invoke(
            {
                "messages": HumanMessage(input("Say: ")),
                "llm": model,
            },
            config,
        )
        for msg in res["messages"]:
            msg.pretty_print()



main()
