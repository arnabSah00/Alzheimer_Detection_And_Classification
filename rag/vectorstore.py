from langchain_chroma import Chroma
from .config import VECTOR_PATH, COLLECTION_NAME

def create_vectorstore(chunks, embeddings):

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_PATH
    )

    db.persist()
    print("✅ Vector DB created")
    return db


def load_vectorstore(embeddings):

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_PATH
    )