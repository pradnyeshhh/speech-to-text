"""Every transcript the server makes, live or uploaded, in one SQLite file."""

import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,        -- unix time
    source TEXT NOT NULL,            -- 'live' (the site's microphone) or 'upload' (POST /transcribe)
    mode TEXT NOT NULL DEFAULT 'dictation',  -- or 'meeting': "[m:ss] You: ..." / "[m:ss] Meeting: ..." lines
    owner TEXT,                      -- the Google email of whoever made it on the subdomain; NULL over the SSH tunnel
    model TEXT NOT NULL,
    audio_s REAL NOT NULL,
    speech_s REAL NOT NULL,
    seconds REAL NOT NULL,           -- upload: decode + transcribe; live: from Stop to the last text
    text TEXT NOT NULL,
    summary TEXT,
    summary_status TEXT NOT NULL,    -- pending | done | error | off (no API key) | none (no text)
    summary_model TEXT,
    summary_error TEXT
)
"""

_COLUMNS = (
    "id, created_at, source, mode, owner, model, audio_s, speech_s, seconds, text,"
    " summary, summary_status, summary_model, summary_error"
)


class Store:
    """Used from the event loop's thread only; each call is one short query."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)  # what people dictate is for the server's owner only
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(transcripts)")}
        if "mode" not in columns:  # a database from before meetings
            self._db.execute("ALTER TABLE transcripts ADD COLUMN mode TEXT NOT NULL DEFAULT 'dictation'")
        if "owner" not in columns:  # a database from before Google sign-in
            self._db.execute("ALTER TABLE transcripts ADD COLUMN owner TEXT")
        self._db.commit()

    def add(self, *, source: str, mode: str, owner: str | None, model: str, audio_s: float, speech_s: float,
            seconds: float, text: str, summary_status: str) -> int:
        cursor = self._db.execute(
            "INSERT INTO transcripts (created_at, source, mode, owner, model, audio_s, speech_s, seconds, text,"
            " summary_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), source, mode, owner, model, round(audio_s, 2), round(speech_s, 2), round(seconds, 2), text,
             summary_status),
        )
        self._db.commit()
        return cursor.lastrowid

    def set_summary(self, transcript_id: int, *, status: str, summary: str | None = None, model: str | None = None,
                    error: str | None = None) -> None:
        self._db.execute(
            "UPDATE transcripts SET summary = ?, summary_status = ?, summary_model = ?, summary_error = ? WHERE id = ?",
            (summary, status, model, error, transcript_id),
        )
        self._db.commit()

    # owner None means everyone's (the SSH tunnel); an email means only that person's.
    _MINE = "(? IS NULL OR owner = ?)"

    def get(self, transcript_id: int, owner: str | None = None) -> dict | None:
        row = self._db.execute(f"SELECT {_COLUMNS} FROM transcripts WHERE id = ? AND {self._MINE}",
                               (transcript_id, owner, owner)).fetchone()
        return dict(row) if row else None

    def recent(self, owner: str | None = None, limit: int = 100) -> list[dict]:
        rows = self._db.execute(f"SELECT {_COLUMNS} FROM transcripts WHERE {self._MINE} ORDER BY id DESC LIMIT ?",
                                (owner, owner, limit))
        return [dict(row) for row in rows]

    def delete(self, transcript_id: int, owner: str | None = None) -> bool:
        cursor = self._db.execute(f"DELETE FROM transcripts WHERE id = ? AND {self._MINE}", (transcript_id, owner, owner))
        self._db.commit()
        return cursor.rowcount > 0

    def stale_pending(self) -> list[int]:
        """Summaries a restart interrupted."""
        return [row[0] for row in self._db.execute("SELECT id FROM transcripts WHERE summary_status = 'pending'")]
