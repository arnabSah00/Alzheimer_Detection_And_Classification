def retrieve_context(db, stage, query, top_k):

    combined_query = f"Alzheimer stage: {stage}. Question: {query}"

    results = db.similarity_search_with_score(combined_query, k=top_k)

    return results


def build_augmented_context(results):

    context_blocks = []
    sources = []

    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        text = doc.page_content.strip()

        context_blocks.append(
            f"Source: {source} | Page: {page} | Chunk: {chunk_id} | Score: {score}\n{text}"
        )

        sources.append({
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
            "score": score,
            "content": text[:300]
        })

    return "\n\n".join(context_blocks), sources