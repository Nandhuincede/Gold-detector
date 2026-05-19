import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Database file path (in project root)
DB_PATH = Path("ornaments.db")


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Allows column access by name
    return conn


def init_db() -> None:
    """
    Initialize the database and create the detections table if it doesn't exist.
    Called on application startup via FastAPI lifespan.
    """
    logger.info(f"Initializing database at {DB_PATH.resolve()}")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT    NOT NULL,
                item_type   TEXT    NOT NULL,
                quantity    INTEGER NOT NULL DEFAULT 0,
                confidence  REAL,
                notes       TEXT,
                created_at  TEXT    NOT NULL
            )
        """)
        # Index for fast history queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_created_at
            ON detections (created_at DESC)
        """)
        conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        conn.close()


def insert_detection(
    filename: str,
    item_type: str,
    quantity: int,
    confidence: Optional[float] = None,
    notes: Optional[str] = None,
) -> int:
    """
    Insert a single ornament detection record into the detections table.

    Args:
        filename:   Original uploaded image filename
        item_type:  Category of ornament (ring, bangle, etc.)
        quantity:   Number of items detected
        confidence: Optional confidence score (0.0 – 1.0)
        notes:      Optional notes about this detection

    Returns:
        The auto-generated row ID of the inserted record.
    """
    created_at = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO detections (filename, item_type, quantity, confidence, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (filename, item_type, quantity, confidence, notes, created_at),
        )
        conn.commit()
        row_id = cursor.lastrowid
        logger.debug(f"Inserted detection id={row_id} for {filename}: {item_type} x{quantity}")
        return row_id
    except Exception as e:
        logger.error(f"Failed to insert detection: {e}")
        raise
    finally:
        conn.close()


def get_recent_detections(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent detection records.

    Args:
        limit: Maximum number of records to return (default 50)

    Returns:
        List of dicts with keys: id, filename, item_type, quantity, confidence, notes, created_at
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, item_type, quantity, confidence, notes, created_at
            FROM detections
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to fetch detections: {e}")
        raise
    finally:
        conn.close()


def get_stats() -> Dict[str, Any]:
    """
    Return aggregate statistics from the detections table.

    Returns:
        Dict with total_records, total_quantity, unique_files, unique_types
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*)            AS total_records,
                COALESCE(SUM(quantity), 0) AS total_quantity,
                COUNT(DISTINCT filename)   AS unique_files,
                COUNT(DISTINCT item_type)  AS unique_types
            FROM detections
        """)
        row = cursor.fetchone()
        return dict(row) if row else {}
    except Exception as e:
        logger.error(f"Failed to fetch stats: {e}")
        raise
    finally:
        conn.close()
