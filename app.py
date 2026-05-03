from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from indexer import index_document
from retriever import retrieve
from generator import generate
from evaluator import evaluate
from database import init_db, log_query
import os
from werkzeug.utils import secure_filename
from scraper import scrape_url
from page_index import get_all_indexed_docs, init_page_index_db

init_page_index_db()

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs("uploads", exist_ok=True)
init_db()

active_doc_id = None
SCORE_THRESHOLD = 3.0
MAX_RETRIES = 2


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    from database import get_all_logs
    logs = get_all_logs()
    return render_template("dashboard.html", logs=logs)


@app.route("/active-doc", methods=["GET"])
def get_active_doc():
    return jsonify({"active_doc_id": active_doc_id})


@app.route("/index-doc", methods=["POST"])
def index_doc():
    global active_doc_id
    data = request.json
    doc_id = data.get("doc_id")
    text = data.get("text")

    if not doc_id or not text:
        return jsonify({"error": "doc_id and text are required"}), 400

    index_document(doc_id, text)
    active_doc_id = doc_id  # ← set active doc on success
    print(f"Active doc set to: {active_doc_id}")
    return jsonify({"status": "success", "doc_id": doc_id})


@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    global active_doc_id

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    doc_id = request.form.get("doc_id", "").strip()

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    if not doc_id:
        doc_id = file.filename.replace(".pdf", "").replace(" ", "-").lower()

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    from indexer import index_pdf
    index_pdf(doc_id, save_path)
    active_doc_id = doc_id  # ← set active doc on success
    print(f"Active doc set to: {active_doc_id}")
    return jsonify({"status": "success", "doc_id": doc_id})


@app.route("/scrape-url", methods=["POST"])
def scrape_url_route():
    global active_doc_id
    data = request.json
    url    = data.get("url", "").strip()
    doc_id = data.get("doc_id", "").strip()

    if not url:
        return jsonify({"error": "URL is required"}), 400

    if not doc_id:
        doc_id = url.replace("https://", "").replace("http://", "").split("/")[0]

    try:
        text = scrape_url(url)
        index_document(doc_id, text)
        word_count = len(text.split())
        active_doc_id = doc_id  # ← set active doc on success (was in except before!)
        print(f"Active doc set to: {active_doc_id}")
        return jsonify({
            "status": "success",
            "doc_id": doc_id,
            "word_count": word_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # ← no active_doc_id set on failure


@app.route("/query", methods=["POST"])
def query():
    global active_doc_id
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "query is required"}), 400

    from indexer import get_indexed_docs, get_partial_matches

    indexed_docs = get_indexed_docs()
    if not indexed_docs:
        return jsonify({
            "answer": None,
            "not_found": True,
            "reason": "no_docs",
            "indexed_docs": [],
            "partial_matches": [],
            "suggestion": "You haven't indexed any documents yet. Paste text, upload a PDF, or scrape a URL to get started."
        })

    print(f"\nQuerying with active_doc_id: '{active_doc_id}'")  # ← debug line

    attempt = 0
    top_k = 5
    answer = None
    scores = None

    while attempt < MAX_RETRIES:
        attempt += 1
        print(f"\n--- Attempt {attempt} (doc: {active_doc_id}) ---")

        # ← THIS is what was missing — pass active_doc_id to retriever
        chunks = retrieve(user_query, top_k=top_k, doc_id=active_doc_id)

        if not chunks:
            break

        answer = generate(user_query, chunks)
        scores = evaluate(user_query, chunks, answer)

        if scores["average_score"] >= SCORE_THRESHOLD:
            print(f"✓ Score {scores['average_score']} passed. Done.")
            break
        else:
            print(f"✗ Score {scores['average_score']} below threshold. Retrying...")
            top_k += 3

    not_found = (scores is None or scores["average_score"] < SCORE_THRESHOLD)

    if not_found:
        partial_matches = get_partial_matches(user_query, doc_id=active_doc_id)
        return jsonify({
            "answer": None,
            "not_found": True,
            "reason": "low_confidence",
            "indexed_docs": indexed_docs,
            "partial_matches": partial_matches,
            "suggestion": f"This topic doesn't appear to be in '{active_doc_id}'. Try indexing a relevant source below."
        })

    log_query(
        query=user_query,
        answer=answer,
        retrieval_relevance=scores["retrieval_relevance"],
        answer_grounding=scores["answer_grounding"],
        answer_quality=scores["answer_quality"],
        average_score=scores["average_score"],
        attempts=attempt
    )

    return jsonify({
        "answer": answer,
        "not_found": False,
        "scores": scores,
        "attempts": attempt,
        "active_doc": active_doc_id
    })

@app.route("/page-index-terms", methods=["GET"])
def page_index_terms():
    """Returns top TF-IDF terms for a given document."""
    from page_index import get_conn
    doc_id = request.args.get("doc_id", "")
    top_k  = int(request.args.get("top_k", 15))

    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400

    conn = get_conn()
    c    = conn.cursor()

    rows = c.execute("""
        SELECT it.term, it.tf, COALESCE(idf.idf, 1.0) as idf,
               it.tf * COALESCE(idf.idf, 1.0) as tfidf
        FROM index_term it
        LEFT JOIN idf_scores idf ON it.term = idf.term
        WHERE it.doc_id = ?
        ORDER BY tfidf DESC
        LIMIT ?
    """, (doc_id, top_k)).fetchall()

    conn.close()
    return jsonify({
        "doc_id": doc_id,
        "terms": [{"term": r[0], "tf": r[1], "idf": r[2], "tfidf": r[3]} for r in rows]
    })


@app.route("/page-index-lookup", methods=["GET"])
def page_index_lookup():
    """Looks up a specific term in the inverted index across all docs."""
    from page_index import get_conn
    term = request.args.get("term", "").strip().lower()

    if not term:
        return jsonify({"error": "term required"}), 400

    conn = get_conn()
    c    = conn.cursor()

    rows = c.execute("""
        SELECT it.doc_id, it.tf, COALESCE(idf.idf, 1.0) as idf,
               it.tf * COALESCE(idf.idf, 1.0) as tfidf,
               json_array_length(it.positions) as positions
        FROM index_term it
        LEFT JOIN idf_scores idf ON it.term = idf.term
        WHERE it.term = ?
        ORDER BY tfidf DESC
    """, (term,)).fetchall()

    conn.close()
    return jsonify({
        "term": term,
        "results": [{"doc_id": r[0], "tf": r[1], "idf": r[2], "tfidf": r[3], "positions": r[4]} for r in rows]
    })

@app.route("/page-index-stats", methods=["GET"])
def page_index_stats():
    docs = get_all_indexed_docs()
    return jsonify({"docs": docs, "total": len(docs)})

@app.route("/api/logs", methods=["GET"])
def api_logs():
    from database import get_all_logs
    try:
        logs = get_all_logs()
        return jsonify({"logs": logs})
    except Exception as e:
        return jsonify({"logs": [], "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)