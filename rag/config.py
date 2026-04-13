import os

BASE_PATH = r"D:\CODE\Alzheimer_Detection_And_Classification\rag"

PDF_PATH = os.path.join(BASE_PATH, "data", "Treatment_doc")
VECTOR_PATH = os.path.join(BASE_PATH, "vector_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "alzheimer_treatment_docs"
TOP_K = 4