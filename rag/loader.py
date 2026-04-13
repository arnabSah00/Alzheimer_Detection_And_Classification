from langchain_community.document_loaders import PyPDFDirectoryLoader
from .config import PDF_PATH

def load_documents():
    loader = PyPDFDirectoryLoader(PDF_PATH, glob="**/*.pdf")
    docs = loader.load()

    print("Loaded documents:", len(docs))
    return docs