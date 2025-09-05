from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent.parent.parent

load_dotenv()


class Settings(BaseSettings):
    CHROMA_DB_FILE_PATH: Path = ROOT_DIR / "chroma"
    CHROMA_DB_FILE_NAME: str = "chroma.sqlite3"
    CHROMA_COLLECTION_NAME: str = "ipf_rules"

    RULES_PDF_PATH: Path = ROOT_DIR / "data" / "rules.pdf"

    EMBEDDING_MODEL: str = "text-embedding-3-large"

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"

    LANGSMITH_MODEL_PROVIDER: str = "openai"
    LANGSMITH_API_KEY: str
    LANGSMITH_TRACING: bool = True
    USER_AGENT: str = "bot"


app_settings = Settings()
