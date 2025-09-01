from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_openai import OpenAIEmbeddings


load_dotenv()


file_path = "./example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

print(
    retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )
)
