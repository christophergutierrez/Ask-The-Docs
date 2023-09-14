"""This script loads documents from a directory, splits them into chunks
and then loads them into a vector store. The vector store is then saved"""
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
)
from langchain.embeddings import (
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import config

EMBEDDING_SOURCE = config.EMBEDDING_SOURCE
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_TOKENS = config.MODEL_TOKENS
VECTOR_STORE = config.VECTOR_STORE
SOURCE_DOCUMENTS = config.SOURCE_DOCUMENTS
OVERLAP = config.OVERLAP
CHARS_PER_TOKEN = config.CHARS_PER_TOKEN


def load_documents(source_dir):
    """Load documents from a soure directory into memory."""
    pdf_loader = DirectoryLoader(
        source_dir,
        glob="./*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True,
    )
    pdf = pdf_loader.load()
    txt_loader = DirectoryLoader(
        source_dir,
        glob="./*.txt",
        loader_cls=TextLoader,
        use_multithreading=True,
    )
    txt = txt_loader.load()
    csv_loader = DirectoryLoader(
        source_dir,
        glob="./*.csv",
        loader_cls=CSVLoader,
        use_multithreading=True,
    )
    csv = csv_loader.load()
    html_loader = DirectoryLoader(
        source_dir,
        glob="./*.html",
        loader_cls=UnstructuredHTMLLoader,
        use_multithreading=True,
    )
    html = html_loader.load()
    all_documents = pdf + txt + html + csv
    return all_documents


print("Loading raw documents into memory and splitting")
documents_raw = load_documents(SOURCE_DOCUMENTS)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHARS_PER_TOKEN * EMBEDDING_TOKENS,
    chunk_overlap=CHARS_PER_TOKEN * OVERLAP,
)
docs = text_splitter.split_documents(documents_raw)

print(f"Loading embedding model {EMBEDDING_MODEL}")
match EMBEDDING_SOURCE:
    case "huggingface":
        embedding = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda"}
        )
    case "openai":
        embedding = OpenAIEmbeddings(
            model=EMBEDDING_MODEL, model_kwargs={"device": "cuda"}
        )
    case _:
        raise ValueError("EMBEDDING_SOURCE must be 'huggingface' or 'openai'")


print("Putting documents in memory as vectors")
db = FAISS.from_documents(docs, embedding)
db.save_local(VECTOR_STORE)
