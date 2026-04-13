import os
from .config import TOP_K, VECTOR_PATH
from .loader import load_documents
from .splitter import split_documents
from .embedding import get_embeddings
from .vectorstore import create_vectorstore, load_vectorstore
from .retriever import retrieve_context, build_augmented_context
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .generator import generate_answer


def build_rag():

    if os.path.exists(VECTOR_PATH):
        print("Vector DB already exists")
        return

    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = get_embeddings()

    create_vectorstore(chunks, embeddings)


def ask_rag(stage, query):

    embeddings = get_embeddings()
    db = load_vectorstore(embeddings)

    results = retrieve_context(db, stage, query, TOP_K)

    context, sources = build_augmented_context(results)

    user_prompt = build_user_prompt(stage, query, context)

    answer = generate_answer(SYSTEM_PROMPT, user_prompt)

    return {
        "answer": answer,
        "sources": sources
    }