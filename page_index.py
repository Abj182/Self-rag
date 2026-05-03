import math
import re
import json
import sqlite3
from collections import defaultdict


# ─── Tokenizer ────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "was", "are", "were",
    "be", "been", "has", "have", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "i", "we", "you", "he", "she", "they", "what", "which", "who",
    "not", "no", "so", "if", "as", "up", "out", "about", "into", "than",
    "then", "when", "there", "their", "they", "also", "just", "more", "over"
}

def tokenize(text):
    """
    Converts raw text into a clean list of tokens.
    Steps: lowercase → remove punctuation → split → remove stopwords
    """
    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


# ─── Database setup ───────────────────────────────────────────────────────────

DB_PATH = "page_index.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_page_index_db():
    """
    Creates 3 tables:
    - documents  : stores each indexed doc (id, name, word count)
    - index_term : the inverted index (term → doc_id, positions, tf score)
    - idf_scores : stores IDF score per term (updated after every new doc)
    """
    conn = get_conn()
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id     TEXT PRIMARY KEY,
            word_count INTEGER,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS index_term (
            term       TEXT,
            doc_id     TEXT,
            tf         REAL,
            positions  TEXT,
            PRIMARY KEY (term, doc_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS idf_scores (
            term TEXT PRIMARY KEY,
            idf  REAL
        )
    """)

    conn.commit()
    conn.close()
    print("Page index DB ready.")


# ─── Indexing ─────────────────────────────────────────────────────────────────

def index_page(doc_id, text):
    """
    Full indexing pipeline for one document:
    1. Tokenize text
    2. Compute TF per term
    3. Store in inverted index with positions
    4. Recompute IDF for all affected terms
    """
    tokens = tokenize(text)
    if not tokens:
        print(f"No tokens found for: {doc_id}")
        return

    total_tokens = len(tokens)

    # build term → positions mapping
    positions = defaultdict(list)
    for pos, token in enumerate(tokens):
        positions[token].append(pos)

    # compute TF = (occurrences of term) / (total tokens in doc)
    tf_scores = {
        term: len(pos_list) / total_tokens
        for term, pos_list in positions.items()
    }

    conn = get_conn()
    c    = conn.cursor()

    # remove old entries for this doc if re-indexing
    c.execute("DELETE FROM documents  WHERE doc_id = ?", (doc_id,))
    c.execute("DELETE FROM index_term WHERE doc_id = ?", (doc_id,))

    # insert document record
    c.execute(
        "INSERT INTO documents (doc_id, word_count) VALUES (?, ?)",
        (doc_id, total_tokens)
    )

    # insert inverted index entries
    for term, pos_list in positions.items():
        c.execute(
            "INSERT INTO index_term (term, doc_id, tf, positions) VALUES (?, ?, ?, ?)",
            (term, doc_id, tf_scores[term], json.dumps(pos_list))
        )

    conn.commit()

    # recompute IDF for every term in this document
    _recompute_idf(c, conn, list(positions.keys()))

    conn.close()
    print(f"Page indexed: '{doc_id}' — {total_tokens} tokens, {len(positions)} unique terms")


def _recompute_idf(cursor, conn, terms):
    """
    IDF = log(total_documents / documents_containing_term)
    Called after every new document is indexed.
    """
    total_docs = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    for term in terms:
        doc_count = cursor.execute(
            "SELECT COUNT(DISTINCT doc_id) FROM index_term WHERE term = ?",
            (term,)
        ).fetchone()[0]

        idf = math.log((total_docs + 1) / (doc_count + 1)) + 1  # smoothed IDF

        cursor.execute(
            "INSERT OR REPLACE INTO idf_scores (term, idf) VALUES (?, ?)",
            (term, idf)
        )

    conn.commit()


# ─── Retrieval ────────────────────────────────────────────────────────────────

def search_index(query, top_k=5, doc_id_filter=None):
    """
    TF-IDF search against the inverted index.

    Steps:
    1. Tokenize the query
    2. For each query token, look up which docs contain it
    3. Score each doc: sum of (TF * IDF) for all matching query terms
    4. Rank docs by score, return top_k

    doc_id_filter: if set, only search within that specific document
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    print(f"\nSearching index for: '{query}' → tokens: {query_tokens}")
    if doc_id_filter:
        print(f"Filtered to doc: {doc_id_filter}")

    conn   = get_conn()
    c      = conn.cursor()
    scores = defaultdict(float)

    for token in query_tokens:
        # get IDF for this term
        row = c.execute(
            "SELECT idf FROM idf_scores WHERE term = ?",
            (token,)
        ).fetchone()

        if not row:
            continue  # term not in index

        idf = row[0]

        # get TF for this term in each document
        if doc_id_filter:
            rows = c.execute(
                "SELECT doc_id, tf FROM index_term WHERE term = ? AND doc_id = ?",
                (token, doc_id_filter)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT doc_id, tf FROM index_term WHERE term = ?",
                (token,)
            ).fetchall()

        for doc_id, tf in rows:
            scores[doc_id] += tf * idf  # TF-IDF score accumulation

    conn.close()

    if not scores:
        print("No matching documents found.")
        return []

    # sort by score descending, return top_k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"Top results: {ranked}")
    return ranked  # list of (doc_id, score)


def get_doc_snippet(doc_id, query, snippet_len=300):
    """
    Retrieves the most relevant snippet from a document for a query.
    Finds the position in the document where query terms appear most densely.
    Used to pass context to the LLM generator.
    """
    conn   = get_conn()
    c      = conn.cursor()
    tokens = tokenize(query)

    # gather all positions of query terms in this doc
    all_positions = []
    for token in tokens:
        row = c.execute(
            "SELECT positions FROM index_term WHERE term = ? AND doc_id = ?",
            (token, doc_id)
        ).fetchone()
        if row:
            all_positions.extend(json.loads(row[0]))

    conn.close()

    if not all_positions:
        return f"[Content from {doc_id}]"

    # find the window of snippet_len tokens with the highest query term density
    all_positions.sort()
    best_start = all_positions[0]
    best_count = 0

    for pos in all_positions:
        count = sum(1 for p in all_positions if pos <= p <= pos + snippet_len)
        if count > best_count:
            best_count  = count
            best_start  = pos

    return f"[From '{doc_id}'] Relevant section contains terms: {', '.join(tokens)}"


def get_all_indexed_docs():
    """Returns list of all doc_ids in the page index."""
    conn  = get_conn()
    c     = conn.cursor()
    rows  = c.execute("SELECT doc_id, word_count FROM documents ORDER BY indexed_at DESC").fetchall()
    conn.close()
    return [{"doc_id": r[0], "word_count": r[1]} for r in rows]