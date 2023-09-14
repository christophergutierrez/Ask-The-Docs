"""This scripts reads from the vector store and allows you to ask questions."""
import textwrap
from typing import Any, Dict
from langchain.chains import RetrievalQA
from langchain.embeddings import (
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings,
)
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
import config

EMBEDDING_SOURCE = config.EMBEDDING_SOURCE
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_TOKENS = config.EMBEDDING_TOKENS
MODEL_NAME = config.MODEL_NAME
MODEL_VERSION = config.MODEL_VERSION
MODEL_TOKENS = config.MODEL_TOKENS
VECTOR_STORE = config.VECTOR_STORE


def wrap_text_preserve_newlines(text: str, width: int = 110) -> str:
    """Formats text to a given width, preserving newlines."""
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = "\n".join(wrapped_lines)
    return wrapped_text


def process_llm_response(llm_response: Dict[str, Any]) -> None:
    """Provide the sources for the response and print the response."""
    print(wrap_text_preserve_newlines(llm_response["result"]))
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


match EMBEDDING_SOURCE:
    case "huggingface":
        print(f"Loading embedding model {EMBEDDING_MODEL}")
        embedding = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda"}
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", revision=MODEL_VERSION
        )
        print("Tokenize")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, model_max_length=EMBEDDING_TOKENS, use_fast=True
        )
        print("Create llm")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MODEL_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    case "openai":
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        llm = ChatOpenAI(model_name=MODEL_NAME)
    case _:
        raise ValueError(f"Unknown embedding source {EMBEDDING_SOURCE}")

db = FAISS.load_local(VECTOR_STORE, embedding)

print("Retrieve & qa_chain")
retriever = db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)


while True:
    query = input("Ask a question: ")
    response = qa_chain(query)
    process_llm_response(response)
