"""SQLite-backed document registry models and persistence helpers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DocumentRecord:
    """One persisted document entry from the registry."""

    doc_id: str
    file_hash: str
    original_filename: str
    stored_path: str
    parser_name: str
    parser_version: str | None
    indexed_status: str
    chunk_count: int | None
    ingested_at: str
    indexed_at: str | None
    last_error: str | None


class DocumentRegistry:
    """Persist and query document metadata in SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    def initialize(self) -> None:
        """Create the registry database schema if it does not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    parser_name TEXT NOT NULL,
                    parser_version TEXT,
                    indexed_status TEXT NOT NULL,
                    chunk_count INTEGER,
                    ingested_at TEXT NOT NULL,
                    indexed_at TEXT,
                    last_error TEXT
                )
                """
            )
            connection.commit()

    def create_document(
            self,
            *,
            doc_id: str,
            file_hash: str,
            original_filename: str,
            stored_path: str,
            parser_name: str,
            parser_version: str | None,
            indexed_status: str,
            ingested_at: str,
            chunk_count: int | None = None,
            indexed_at: str | None = None,
            last_error: str | None = None,
    ) -> None:
        """Insert one new document record."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    doc_id,
                    file_hash,
                    original_filename,
                    stored_path,
                    parser_name,
                    parser_version,
                    indexed_status,
                    chunk_count,
                    ingested_at,
                    indexed_at,
                    last_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    file_hash,
                    original_filename,
                    stored_path,
                    parser_name,
                    parser_version,
                    indexed_status,
                    chunk_count,
                    ingested_at,
                    indexed_at,
                    last_error,
                ),
            )
            connection.commit()

    def mark_indexed(
            self,
            *,
            doc_id: str,
            chunk_count: int,
            indexed_at: str,
    ) -> None:
        """Mark a document as indexed and store its chunk count."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET indexed_status = ?, chunk_count = ?, indexed_at = ?, last_error = NULL
                WHERE doc_id = ?
                """,
                (
                    "indexed",
                    chunk_count,
                    indexed_at,
                    doc_id,
                ),
            )
            connection.commit()

    def mark_failed(
            self,
            *,
            doc_id: str,
            error_message: str,
    ) -> None:
        """Mark a document as failed and store the last error."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET indexed_status = ?, last_error = ?
                WHERE doc_id = ?
                """,
                (
                    "failed",
                    error_message,
                    doc_id,
                ),
            )
            connection.commit()

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """Return one document record by id."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    doc_id,
                    file_hash,
                    original_filename,
                    stored_path,
                    parser_name,
                    parser_version,
                    indexed_status,
                    chunk_count,
                    ingested_at,
                    indexed_at,
                    last_error
                FROM documents
                WHERE doc_id = ?
                """,
                (doc_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def list_documents(self) -> list[DocumentRecord]:
        """Return all document records ordered by most recent ingest time."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    doc_id,
                    file_hash,
                    original_filename,
                    stored_path,
                    parser_name,
                    parser_version,
                    indexed_status,
                    chunk_count,
                    ingested_at,
                    indexed_at,
                    last_error
                FROM documents
                ORDER BY ingested_at DESC
                """
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def delete_document(self, doc_id: str) -> None:
        """Delete one document record from the registry."""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE doc_id = ?",
                (doc_id,),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row access by column name."""
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        """Convert a SQLite row into a typed document record."""
        return DocumentRecord(
            doc_id=row["doc_id"],
            file_hash=row["file_hash"],
            original_filename=row["original_filename"],
            stored_path=row["stored_path"],
            parser_name=row["parser_name"],
            parser_version=row["parser_version"],
            indexed_status=row["indexed_status"],
            chunk_count=row["chunk_count"],
            ingested_at=row["ingested_at"],
            indexed_at=row["indexed_at"],
            last_error=row["last_error"],
        )

    def mark_pending(self, *, doc_id: str) -> None:
        """Mark a document as pending reindexing."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET indexed_status = ?, last_error = NULL
                WHERE doc_id = ?
                """,
                ("pending", doc_id),
            )
            connection.commit()
