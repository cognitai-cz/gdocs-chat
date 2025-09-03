import logging

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import app_settings


class PDFEmbedding:
    def __init__(self) -> None:
        self.pdf_file = app_settings.RULES_PDF_PATH
        self.chroma_file = app_settings.CHROMA_DB_FILE_PATH
        self.embeddings = OpenAIEmbeddings(model=app_settings.EMBEDDING_MODEL)
        self.logger = logging.getLogger(__name__)

    def load_to_db(self):
        if not (self.chroma_file / app_settings.CHROMA_DB_FILE_NAME).exists():
            self.logger.info("Creating embeddings")
            split_documents = self._split_pdf()
            self.vector_store.add_documents(split_documents)
        else:
            self.logger.info("Embeddings already exist, skipping")

    @property
    def vector_store(self):
        return Chroma(
            collection_name=app_settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=app_settings.CHROMA_DB_FILE_PATH,
        )

    def _split_pdf(self) -> list[Document]:
        loader = PyPDFLoader(self.pdf_file)
        content = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        return text_splitter.split_documents(content)

