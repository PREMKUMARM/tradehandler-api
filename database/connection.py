"""
SQLite database connection and setup utilities
"""
import sqlite3
import os
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import threading


class DatabaseConnection:
    """SQLite database connection manager - thread-safe"""

    def __init__(self, db_path: str = "tradehandler.db"):
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
        """Create all necessary tables"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Agent Approvals Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_approvals (
                approval_id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                details TEXT NOT NULL,  -- JSON
                trade_value REAL NOT NULL,
                risk_amount REAL NOT NULL,
                reward_amount REAL DEFAULT 0.0,
                risk_percentage REAL NOT NULL,
                rr_ratio REAL DEFAULT 0.0,
                reasoning TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                approved_at TEXT,
                rejected_at TEXT,
                approved_by TEXT,
                rejected_by TEXT,
                rejection_reason TEXT,
                symbol TEXT,
                entry_price REAL,
                quantity INTEGER,
                stop_loss REAL,
                target_price REAL,
                entry_order_id TEXT,
                sl_order_id TEXT,
                tp_order_id TEXT
            )
        ''')

        # Agent Logs Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                component TEXT NOT NULL,
                metadata TEXT  -- JSON
            )
        ''')

        # Agent Config Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                value_type TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                updated_at TEXT NOT NULL
            )
        ''')

        # Simulation Results Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_results (
                simulation_id TEXT PRIMARY KEY,
                instrument_name TEXT NOT NULL,
                date_range TEXT NOT NULL,
                strategy TEXT NOT NULL,
                trades TEXT NOT NULL,  -- JSON
                summary TEXT NOT NULL,  -- JSON
                created_at TEXT NOT NULL,
                file_path TEXT
            )
        ''')

        # Tool Executions Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tool_executions (
                execution_id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                inputs TEXT NOT NULL,  -- JSON
                outputs TEXT NOT NULL,  -- JSON
                execution_time REAL NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                timestamp TEXT NOT NULL
            )
        ''')

        # Chat Messages Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT  -- JSON
            )
        ''')

        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_logs_timestamp ON agent_logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_logs_component ON agent_logs(component)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_approvals_status ON agent_approvals(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_approvals_symbol ON agent_approvals(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tool_executions_tool ON tool_executions(tool_name)')

        # Migration: Add order ID columns if they don't exist
        try:
            cursor.execute("ALTER TABLE agent_approvals ADD COLUMN entry_order_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE agent_approvals ADD COLUMN sl_order_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE agent_approvals ADD COLUMN tp_order_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        conn.commit()

    def execute_query(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query and return cursor"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor

    def execute_many(self, query: str, params_list: list) -> sqlite3.Cursor:
        """Execute many queries"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        return cursor

    def commit(self):
        """Commit changes"""
        conn = self.get_connection()
        conn.commit()


# Global database instance
_db_instance: Optional[DatabaseConnection] = None


def get_database() -> DatabaseConnection:
    """Get global database instance"""
    global _db_instance
    if _db_instance is None:
        db_path = os.getenv("DATABASE_PATH", "data/tradehandler.db")
        _db_instance = DatabaseConnection(db_path)
    return _db_instance


def init_database():
    """Initialize database and create tables"""
    db = get_database()
    db.get_connection()  # This triggers table creation
    return db
