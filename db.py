import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS batches(
 id INTEGER PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, record_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS feedback(
 id INTEGER PRIMARY KEY, batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
 feedback_date TEXT, source TEXT NOT NULL, body TEXT NOT NULL,
 topic_auto TEXT NOT NULL DEFAULT '待分析', topic_manual TEXT,
 sentiment_auto TEXT NOT NULL DEFAULT 'neutral', sentiment_manual TEXT,
 sentiment_terms_auto TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_feedback_batch ON feedback(batch_id);
CREATE TABLE IF NOT EXISTS analysis_runs(
 id INTEGER PRIMARY KEY, batch_id INTEGER NOT NULL UNIQUE REFERENCES batches(id) ON DELETE CASCADE,
 cluster_count INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS topic_summaries(
 id INTEGER PRIMARY KEY, analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
 topic_id INTEGER NOT NULL, label_auto TEXT NOT NULL, count INTEGER NOT NULL,
 negative_share REAL NOT NULL, recent_growth REAL, priority_score REAL NOT NULL,
 score_components_json TEXT NOT NULL, top_terms_json TEXT NOT NULL,
 UNIQUE(analysis_run_id, topic_id)
);
"""
def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(feedback)")}
    if "sentiment_terms_auto" not in columns:
        connection.execute("ALTER TABLE feedback ADD COLUMN sentiment_terms_auto TEXT NOT NULL DEFAULT '[]'")
    if "rating" not in columns:
        connection.execute("ALTER TABLE feedback ADD COLUMN rating INTEGER")
