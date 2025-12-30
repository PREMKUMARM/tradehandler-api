"""
SQLite database connection for selected stocks
"""
import sqlite3
import os
from pathlib import Path
from typing import Optional
import threading


class StocksDatabaseConnection:
    """SQLite database connection manager for stocks - thread-safe"""

    def __init__(self, db_path: str = "algofeast_stocks.db"):
        self.db_path = db_path
        self._connections = {}  # Thread-local connections
        self._lock = threading.Lock()
        self._tables_created = False
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """Ensure the database directory exists"""
        db_dir = Path(self.db_path).parent
        if db_dir and not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection with row factory"""
        thread_id = threading.get_ident()

        if thread_id not in self._connections:
            # Create a new connection for this thread
            self._connections[thread_id] = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connections[thread_id].row_factory = sqlite3.Row

            # Create tables if not already done
            if not self._tables_created:
                with self._lock:
                    if not self._tables_created:
                        self._create_tables()
                        self._tables_created = True

        return self._connections[thread_id]

    def close(self):
        """Close all thread-local database connections"""
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass  # Ignore errors when closing
        self._connections.clear()

    def _create_tables(self):
        """Create all necessary tables for stocks"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Selected Stocks Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS selected_stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tradingsymbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                instrument_token INTEGER NOT NULL,
                instrument_key TEXT NOT NULL UNIQUE,
                name TEXT,
                instrument_type TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                notes TEXT,
                UNIQUE(tradingsymbol, exchange)
            )
        ''')

        # Create indexes for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_stocks_instrument_key 
            ON selected_stocks(instrument_key)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_stocks_tradingsymbol 
            ON selected_stocks(tradingsymbol)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_stocks_active 
            ON selected_stocks(is_active)
        ''')

        conn.commit()

    def execute_query(self, query: str, params: tuple = ()):
        """Execute a query and return cursor"""
        conn = self.get_connection()
        return conn.execute(query, params)

    def commit(self):
        """Commit the current transaction"""
        conn = self.get_connection()
        conn.commit()


# Global instance
_stocks_db: Optional[StocksDatabaseConnection] = None


def get_stocks_database() -> StocksDatabaseConnection:
    """Get the global stocks database instance"""
    global _stocks_db
    if _stocks_db is None:
        _stocks_db = StocksDatabaseConnection()
    return _stocks_db

