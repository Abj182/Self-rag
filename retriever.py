from indexer import model, collection
from page_index import search_index, get_doc_snippet


def retrieve(query, top_k=5, doc_id=None):
    """
    Two-stage retrieval:
    Stage 1 — TF-IDF inverted index finds the most relevant documents
    Stage 2 — ChromaDB fetches the best matching chunks from those docs

    This matches your project spec exactly:
    Inverted Index + TF-IDF for ranking, vector search for chunk extraction.
    """
    print(f"\nRetrieving for: '{query}'" + (f" [doc: {doc_id}]" if doc_id else ""))

    # ── Stage 1: TF-IDF ranking ──────────────────────────────────────────────
    tfidf_results = search_index(query, top_k=top_k, doc_id_filter=doc_id)

    if not tfidf_results:
        print("TF-IDF found nothing. Falling back to vector search.")
        return _vector_search(query, top_k, doc_id)

    # get the top ranked doc IDs from TF-IDF
    top_doc_ids = [doc_id for doc_id, score in tfidf_results]
    print(f"TF-IDF top docs: {top_doc_ids}")

    # ── Stage 2: vector search within top-ranked docs ────────────────────────
    all_chunks = []

    for ranked_doc_id in top_doc_ids[:3]:  # top 3 docs from TF-IDF
        chunks = _vector_search(query, top_k=3, doc_id=ranked_doc_id)
        all_chunks.extend(chunks)

    if not all_chunks:
        # fallback: return snippet summaries from TF-IDF results
        all_chunks = [get_doc_snippet(d, query) for d, _ in tfidf_results]

    print(f"Final chunks retrieved: {len(all_chunks)}")
    return all_chunks


def _vector_search(query, top_k=5, doc_id=None):
    """
    Pure ChromaDB vector search.
    Used as Stage 2 after TF-IDF narrows down the docs.
    """
    try:
        query_embedding = model.encode([query]).tolist()

        if doc_id:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where={"doc_id": doc_id}
            )
        else:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )

        chunks    = results["documents"][0]
        distances = results["distances"][0]

        print(f"Vector search found {len(chunks)} chunks:")
        for i, (chunk, dist) in enumerate(zip(chunks, distances)):
            print(f"  [{i+1}] score={dist:.4f} | {chunk[:80]}...")

        return chunks

    except Exception as e:
        print(f"Vector search error: {e}")
        return []