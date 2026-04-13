from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    # add chunk_id (IMPORTANT FROM PDF)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk-{i+1}"

    print("Total chunks:", len(chunks))
    return chunks