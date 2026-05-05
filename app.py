from dotenv import load_dotenv
load_dotenv()

"""
app.py - MCQ Generator Backend
"""

import os
import re
import json
import random
import sqlite3
import logging
from functools import lru_cache
from flask import Flask, request, jsonify, render_template
from openai import OpenAI, OpenAIError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH     = os.getenv("DB_PATH", "mcq.db")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY", "")
MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS  = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))   # chars, not tokens
MAX_CHUNKS  = int(os.getenv("MAX_CHUNKS", "10"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app    = Flask(__name__)
client = OpenAI(api_key=OPENAI_KEY)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with get_db() as conn:
        return conn.execute(sql, params).fetchall()

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_NOISE = re.compile(
    r"""(
        \b\d{1,3}\b                    |  # standalone page numbers
        [\u2022\u2023\u25E6\u2043\u2219]  |  # bullet symbols
        [©®™°•·–—‒]                    |  # misc symbols
        \b(fig|figure|table|equation|appendix|chapter|section|exercise)\b  # layout refs
    )""",
    re.IGNORECASE | re.VERBOSE,
)

def clean_text(text: str) -> str:
    text = _NOISE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noisy(text: str) -> bool:
    """True if page has too little real content."""
    words = re.findall(r"[a-zA-Z]{3,}", text)
    return len(words) < 30


def build_context(chunks: list, max_chars: int = MAX_TOKENS) -> str:
    random.shuffle(chunks)
    selected, total = [], 0
    for chunk in chunks:
        cleaned = clean_text(chunk["content"])
        if is_noisy(cleaned):
            continue
        if total + len(cleaned) > max_chars:
            break
        selected.append(cleaned)
        total += len(cleaned)
    return "\n\n---\n\n".join(selected)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert educational question paper setter for school and competitive exams.
Your task is to generate high-quality questions from the provided textbook content.

STRICT RULES:
- Questions must test CONCEPTUAL UNDERSTANDING, not rote recall of text.
- NEVER reference page numbers, figure labels, table numbers, or formatting.
- NEVER generate duplicate or similar questions.
- Cover a DIVERSE range of topics from the provided content.
- Options for MCQs must be plausible but clearly distinguishable.
- Correct answer must always be one of the four options.
- Difficulty: mix easy, medium, and hard questions proportionally.
- Language: clear, grammatically correct, academically appropriate.
- DO NOT add any explanation or commentary outside the JSON.

OUTPUT FORMAT (strict JSON, nothing else):
{
  "mcq": [
    {"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "A. ..."}
  ],
  "two_marks": [{"question": "..."}],
  "three_marks": [{"question": "..."}],
  "five_marks": [{"question": "..."}],
  "ten_marks": [{"question": "..."}]
}

Counts: mcq=10, two_marks=5, three_marks=3, five_marks=3, ten_marks=2
"""

def make_user_prompt(subject: str, context: str) -> str:
    return f"""Subject: {subject}

Textbook Content:
\"\"\"
{context}
\"\"\"

Generate questions exactly as specified. Return ONLY valid JSON."""


def call_llm(subject: str, context: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_user_prompt(subject, context)},
        ],
        temperature=0.7,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def validate_output(data: dict) -> dict:
    """Ensure all required keys exist and have correct structure."""
    schema = {
        "mcq":         {"count": 10, "has_options": True},
        "two_marks":   {"count": 5,  "has_options": False},
        "three_marks": {"count": 3,  "has_options": False},
        "five_marks":  {"count": 3,  "has_options": False},
        "ten_marks":   {"count": 2,  "has_options": False},
    }
    result = {}
    for key, rules in schema.items():
        items = data.get(key, [])
        if not isinstance(items, list):
            items = []
        # Validate each item has "question"
        valid = [q for q in items if isinstance(q, dict) and q.get("question")]
        result[key] = valid[:rules["count"]]
    return result

# ---------------------------------------------------------------------------
# Routes — metadata
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/boards")
def boards():
    rows = query("SELECT DISTINCT board FROM chunks ORDER BY board")
    return jsonify([r["board"] for r in rows])


@app.route("/classes")
def classes():
    board = request.args.get("board", "").strip()
    if not board:
        return jsonify({"error": "board is required"}), 400
    rows = query(
        "SELECT DISTINCT class FROM chunks WHERE board=? ORDER BY class",
        (board,)
    )
    return jsonify([r["class"] for r in rows])


@app.route("/subjects_by_class")
def subjects_by_class():
    board = request.args.get("board", "").strip()
    cls   = request.args.get("class", "").strip()
    if not board or not cls:
        return jsonify({"error": "board and class are required"}), 400
    rows = query(
        "SELECT DISTINCT subject FROM chunks WHERE board=? AND class=? ORDER BY subject",
        (board, cls)
    )
    return jsonify([r["subject"] for r in rows])


@app.route("/books_by_subject")
def books_by_subject():
    board   = request.args.get("board", "").strip()
    cls     = request.args.get("class", "").strip()
    subject = request.args.get("subject", "").strip()
    if not all([board, cls, subject]):
        return jsonify({"error": "board, class, and subject are required"}), 400
    rows = query(
        "SELECT DISTINCT book FROM chunks WHERE board=? AND class=? AND subject=? ORDER BY book",
        (board, cls, subject)
    )
    return jsonify([r["book"] for r in rows])


@app.route("/page_range")
def page_range():
    board   = request.args.get("board", "").strip()
    cls     = request.args.get("class", "").strip()
    subject = request.args.get("subject", "").strip()
    book    = request.args.get("book", "").strip()
    if not all([board, cls, subject, book]):
        return jsonify({"error": "all filters required"}), 400
    rows = query(
        "SELECT MIN(page) as min_page, MAX(page) as max_page FROM chunks "
        "WHERE board=? AND class=? AND subject=? AND book=?",
        (board, cls, subject, book)
    )
    if not rows or rows[0]["min_page"] is None:
        return jsonify({"error": "no pages found"}), 404
    return jsonify({"min": rows[0]["min_page"], "max": rows[0]["max_page"]})

# ---------------------------------------------------------------------------
# Route — generate MCQ
# ---------------------------------------------------------------------------

@app.route("/generate-mcq", methods=["POST"])
def generate_mcq():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "JSON body required"}), 400

    board      = body.get("board", "").strip()
    cls        = body.get("class", "").strip()
    subject    = body.get("subject", "").strip()
    book       = body.get("book", "").strip()
    page_start = body.get("page_start")
    page_end   = body.get("page_end")

    if not all([board, cls, subject, book]):
        return jsonify({"error": "board, class, subject, and book are required"}), 400

    # Build query
    sql    = "SELECT * FROM chunks WHERE board=? AND class=? AND subject=? AND book=?"
    params = [board, cls, subject, book]

    if page_start is not None and page_end is not None:
        try:
            ps, pe = int(page_start), int(page_end)
            if ps > pe or ps < 1:
                return jsonify({"error": "Invalid page range"}), 400
            sql    += " AND page BETWEEN ? AND ?"
            params += [ps, pe]
        except (ValueError, TypeError):
            return jsonify({"error": "page_start and page_end must be integers"}), 400

    rows = query(sql, tuple(params))
    if not rows:
        return jsonify({"error": "No content found for the given selection"}), 404

    # Sample up to MAX_CHUNKS
    sample = random.sample(list(rows), min(MAX_CHUNKS, len(rows)))
    context = build_context(sample)

    if len(context) < 200:
        return jsonify({"error": "Insufficient content to generate questions"}), 422

    try:
        raw_output = call_llm(subject, context)
        result     = validate_output(raw_output)
    except OpenAIError as e:
        log.error(f"OpenAI error: {e}")
        return jsonify({"error": f"LLM API error: {str(e)}"}), 502
    except (json.JSONDecodeError, KeyError) as e:
        log.error(f"Parse error: {e}")
        return jsonify({"error": "Failed to parse LLM response. Try again."}), 500

    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not OPENAI_KEY:
        log.warning("OPENAI_API_KEY is not set!")
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", port=5000)