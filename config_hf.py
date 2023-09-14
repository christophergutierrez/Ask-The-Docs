"""This is where all variables are stored for ingest and ask
Below is an example that uses HuggingFace's
This is 100% local, no internet connection required, but needs GPU"""
from typing import Optional

# Used by ingest.py and ask.py
EMBEDDING_SOURCE: str = "huggingface"
EMBEDDING_MODEL: str = "impira/layoutlm-document-qa"
EMBEDDING_TOKENS: int = 512

# Used by ingest.py
SOURCE_DOCUMENTS: str = "./source_data/example/"
VECTOR_STORE: str = "./faiss_dbs/example.faiss"
OVERLAP: int = 200
CHARS_PER_TOKEN: int = 4

# Used by ask.py
MODEL_NAME: str = "TheBloke/orca_mini_7B-GPTQ"
MODEL_VERSION: Optional[str] = "main"
MODEL_TOKENS: int = 2048
