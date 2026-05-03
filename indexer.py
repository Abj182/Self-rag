from sentence_transformers import SentenceTransformer
import chromadb
import fitz
import os
from page_index import index_page, init_page_index_db

init_page_index_db()
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("pages")


def chunk_text(text, size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += size - overlap
    return chunks


def index_document(doc_id, text):
    chunks = chunk_text(text)
    if not chunks:
        print(f"No chunks found for doc: {doc_id}")
        return

    print(f"Indexing '{doc_id}' → {len(chunks)} chunks...")

    # delete old chunks for this doc_id if it was indexed before
    try:
        existing = collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            collection.delete(where={"doc_id": doc_id})
            print(f"Replaced existing '{doc_id}' chunks.")
    except Exception:
        pass

    embeddings = model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"doc_id": doc_id} for _ in chunks]  # ← key fix
    )
    index_page(doc_id, text)

    print(f"Done. '{doc_id}' indexed with metadata.")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n[Page {page_num + 1}]\n{text}"
    doc.close()
    print(f"Extracted {len(full_text.split())} words from PDF.")
    return full_text


def index_pdf(doc_id, pdf_path):
    text = extract_text_from_pdf(pdf_path)
    index_document(doc_id, text)


def get_indexed_docs():
    try:
        results = collection.get()
        ids = results.get("ids", [])
        doc_ids = set()
        for chunk_id in ids:
            parts = chunk_id.rsplit("_chunk_", 1)
            if len(parts) == 2:
                doc_ids.add(parts[0])
        return sorted(list(doc_ids))
    except Exception:
        return []


def get_partial_matches(query, doc_id=None, top_k=3):
    try:
        query_embedding = model.encode([query]).tolist()
        where = {"doc_id": doc_id} if doc_id else None
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where
        )
        chunks = results["documents"][0]
        return [c[:120].strip() for c in chunks if c.strip()]
    except Exception:
        return []