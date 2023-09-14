"""This is where all variables are stored for ingest and ask
Below is an example that uses OpenAI's API
This sends data to the OpenAI API, so you need an internet connection"""
from typing import Optional

# Used by ingest.py and ask.py
EMBEDDING_SOURCE: str = "openai"
EMBEDDING_MODEL: str = "text-embedding-ada-002"
EMBEDDING_TOKENS: int = 512

# Used by ingest.py
SOURCE_DOCUMENTS: str = "./source_data/pets/"
VECTOR_STORE: str = "./faiss_dbs/pets.faiss"
OVERLAP: int = 200
CHARS_PER_TOKEN: int = 4

# Used by ask.py
MODEL_NAME: str = "gpt-3.5-turbo"
MODEL_VERSION: Optional[str] = None
MODEL_TOKENS: int = 2048
