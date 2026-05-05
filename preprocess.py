"""
preprocess.py - Extract text from PDFs and store page-wise in SQLite.
Usage: python preprocess.py [--books-dir books] [--db mcq.db]
"""

import os
import sys
import sqlite3
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            board   TEXT NOT NULL,
            class   TEXT NOT NULL,
            subject TEXT NOT NULL,
            book    TEXT NOT NULL,
            page    INTEGER NOT NULL,
            content TEXT NOT NULL,
            UNIQUE(board, class, subject, book, page)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_lookup ON chunks(board, class, subject, book)")
    conn.commit()
    return conn


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return list of (page_number, text) tuples (1-indexed)."""
    try:
        import pdfplumber
    except ImportError:
        log.error("pdfplumber not installed. Run: pip install pdfplumber")
        sys.exit(1)

    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if len(text) > 50:          # skip near-empty pages
                    pages.append((i, text))
    except Exception as e:
        log.warning(f"Failed to read {pdf_path}: {e}")
    return pages


def ingest_books(books_dir: str, db_path: str) -> None:
    conn = init_db(db_path)
    root = Path(books_dir)

    if not root.exists():
        log.error(f"Books directory not found: {books_dir}")
        sys.exit(1)

    inserted = skipped = 0

    # Expected hierarchy: books/<BOARD>/<CLASS>/<SUBJECT>/<book>.pdf
    for pdf_path in sorted(root.rglob("*.pdf")):
        parts = pdf_path.relative_to(root).parts
        if len(parts) != 4:
            log.warning(f"Skipping (unexpected path depth): {pdf_path}")
            continue

        board, cls, subject, filename = parts
        book = Path(filename).stem

        log.info(f"Processing: {board}/{cls}/{subject}/{book}")
        pages = extract_pages(str(pdf_path))

        for page_num, content in pages:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO chunks (board, class, subject, book, page, content) VALUES (?,?,?,?,?,?)",
                    (board, cls, subject, book, page_num, content),
                )
                inserted += 1
            except sqlite3.Error as e:
                log.warning(f"DB error on {book} p{page_num}: {e}")

        conn.commit()
        skipped_pages = len(pages) - inserted if inserted else 0
        log.info(f"  → {len(pages)} pages extracted")

    conn.close()
    log.info(f"\nDone. {inserted} pages inserted (duplicates skipped automatically).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDF books into SQLite")
    parser.add_argument("--books-dir", default="books", help="Root books directory")
    parser.add_argument("--db", default="mcq.db", help="SQLite database path")
    args = parser.parse_args()
    ingest_books(args.books_dir, args.db)