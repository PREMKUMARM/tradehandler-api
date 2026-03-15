"""
Agent Storage Layer using SQLite/ChromaDB
"""

import sqlite3
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from pathlib import Path

# ChromaDB for vector storage
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available, using SQLite only")

from utils.logger import log_info, log_error, log_warning

class AgentDataStore:
    """Data store for agent data using SQLite and ChromaDB"""
    
    def __init__(self, db_path: str = "agent_data.db", chroma_path: str = "./chroma_db"):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.db_connection = None
        self.chroma_client = None
        self.collections = {}
        
    async def initialize(self):
        """Initialize the data store"""
        
        # Initialize SQLite
        await self._init_sqlite()
        
        # Initialize ChromaDB if available
        if CHROMADB_AVAILABLE:
            await self._init_chromadb()
        
        log_info("Agent data store initialized")
    
    async def _init_sqlite(self):
        """Initialize SQLite database"""
        
        try:
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.db_connection.row_factory = sqlite3.Row
            
            # Create tables
            await self._create_tables()
            
            log_info("SQLite database initialized")
            
        except Exception as e:
            log_error(f"Failed to initialize SQLite: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables"""
        
        cursor = self.db_connection.cursor()
        
        # Agent data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_data (
                agent_id TEXT PRIMARY KEY,
                agent_type TEXT,
                name TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Agent tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_tasks (
                task_id TEXT PRIMARY KEY,
                agent_id TEXT,
                task_type TEXT,
                status TEXT,
                data TEXT,
                result TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                execution_time REAL
            )
        """)
        
        # Agent communication table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_communication (
                message_id TEXT PRIMARY KEY,
                from_agent TEXT,
                to_agent TEXT,
                message_type TEXT,
                data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Strategies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                components TEXT,
                risk_parameters TEXT,
                timeframes TEXT,
                instruments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                backtest_results TEXT,
                optimization_results TEXT
            )
        """)
        
        # Pre-market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS premarket_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
        """)
        
        # Agent metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_metrics (
                agent_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (agent_id, metric_name, timestamp)
            )
        """)
        
        # Agent configurations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_configurations (
                agent_id TEXT PRIMARY KEY,
                configuration TEXT,
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_connection.commit()
    
    async def _init_chromadb(self):
        """Initialize ChromaDB for vector storage"""
        
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            
            # Create collections
            self.collections["agent_embeddings"] = self.chroma_client.get_or_create_collection(
                name="agent_embeddings",
                metadata={"description": "Agent performance embeddings"}
            )
            
            self.collections["strategy_embeddings"] = self.chroma_client.get_or_create_collection(
                name="strategy_embeddings",
                metadata={"description": "Strategy performance embeddings"}
            )
            
            self.collections["market_embeddings"] = self.chroma_client.get_or_create_collection(
                name="market_embeddings",
                metadata={"description": "Market condition embeddings"}
            )
            
            log_info("ChromaDB initialized successfully")
            
        except Exception as e:
            log_error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
    
    # Agent Data Operations
    async def save_agent_data(self, agent_id: str, data: Dict[str, Any]):
        """Save agent data"""
        
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO agent_data (agent_id, agent_type, name, data, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            agent_id,
            data.get("agent_type", ""),
            data.get("name", ""),
            json.dumps(data),
            datetime.now()
        ))
        
        self.db_connection.commit()
    
    async def get_agent_data(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent data"""
        
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM agent_data WHERE agent_id = ?", (agent_id,))
        row = cursor.fetchone()
        
        if row:
            return json.loads(row["data"])
        return None
    
    async def get_all_agents_data(self) -> List[Dict[str, Any]]:
        """Get all agents data"""
        
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM agent_data ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        
        return [json.loads(row["data"]) for row in rows]
    
    # Task Operations
    async def save_task(self, task: Dict[str, Any]):
        """Save task data"""
        
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO agent_tasks 
            (task_id, agent_id, task_type, status, data, result, error, completed_at, execution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.get("task_id"),
            task.get("agent_id"),
            task.get("task_type"),
            task.get("status"),
            json.dumps(task.get("data", {})),
            json.dumps(task.get("result", {})),
            task.get("error"),
            task.get("completed_at"),
            task.get("execution_time")
        ))
        
        self.db_connection.commit()
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task data"""
        
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM agent_tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                "task_id": row["task_id"],
                "agent_id": row["agent_id"],
                "task_type": row["task_type"],
                "status": row["status"],
                "data": json.loads(row["data"]),
                "result": json.loads(row["result"]),
                "error": row["error"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "execution_time": row["execution_time"]
            }
        return None
    
    async def get_agent_tasks(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get tasks for an agent"""
        
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT * FROM agent_tasks WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
            (agent_id, limit)
        )
        rows = cursor.fetchall()
        
        return [
            {
                "task_id": row["task_id"],
                "agent_id": row["agent_id"],
                "task_type": row["task_type"],
                "status": row["status"],
                "data": json.loads(row["data"]),
                "result": json.loads(row["result"]),
                "error": row["error"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "execution_time": row["execution_time"]
            }
            for row in rows
        ]
    
    # Communication Operations
    async def save_message(self, message: Dict[str, Any]):
        """Save agent communication message"""
        
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
            INSERT INTO agent_communication 
            (message_id, from_agent, to_agent, message_type, data, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            message.get("message_id"),
            message.get("from_agent"),
            message.get("to_agent"),
            message.get("message_type"),
            json.dumps(message.get("data", {})),
            message.get("timestamp", datetime.now())
        ))
        
        self.db_connection.commit()
    
    async def get_agent_messages(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages for an agent"""
        
        cursor = self.db_connection.cursor()
        cursor.execute("""
            SELECT * FROM agent_communication 
            WHERE from_agent = ? OR to_agent = ? 
            ORDER BY timestamp DESC LIMIT ?
        """, (agent_id, agent_id, limit))
        rows = cursor.fetchall()
        
        return [
            {
                "message_id": row["message_id"],
                "from_agent": row["from_agent"],
                "to_agent": row["to_agent"],
                "message_type": row["message_type"],
                "data": json.loads(row["data"]),
                "timestamp": row["timestamp"],
                "processed": row["processed"]
            }
            for row in rows
        ]
    
    # Strategy Operations
    async def save_strategy(self, strategy: Dict[str, Any]):
        """Save strategy data"""
        
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO strategies 
            (strategy_id, name, description, components, risk_parameters, timeframes, instruments, backtest_results, optimization_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy.get("strategy_id"),
            strategy.get("name"),
            strategy.get("description"),
            json.dumps(strategy.get("components", [])),
            json.dumps(strategy.get("risk_parameters", {})),
            json.dumps(strategy.get("timeframes", [])),
            json.dumps(strategy.get("instruments", [])),
            json.dumps(strategy.get("backtest_results", {})),
            json.dumps(strategy.get("optimization_results", {}))
        ))
        
        self.db_connection.commit()
        
        # Also save to ChromaDB if available
        if self.chroma_client and "strategy_embeddings" in self.collections:
            await self._save_strategy_embedding(strategy)
    
    async def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy data"""
        
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                "strategy_id": row["strategy_id"],
                "name": row["name"],
                "description": row["description"],
                "components": json.loads(row["components"]),
                "risk_parameters": json.loads(row["risk_parameters"]),
                "timeframes": json.loads(row["timeframes"]),
                "instruments": json.loads(row["instruments"]),
                "created_at": row["created_at"],
                "backtest_results": json.loads(row["backtest_results"]),
                "optimization_results": json.loads(row["optimization_results"])
            }
        return None
    
    async def get_all_strategies(self) -> List[Dict[str, Any]]:
        """Get all strategies"""
        
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM strategies ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        return [
            {
                "strategy_id": row["strategy_id"],
                "name": row["name"],
                "description": row["description"],
                "components": json.loads(row["components"]),
                "risk_parameters": json.loads(row["risk_parameters"]),
                "timeframes": json.loads(row["timeframes"]),
                "instruments": json.loads(row["instruments"]),
                "created_at": row["created_at"],
                "backtest_results": json.loads(row["backtest_results"]),
                "optimization_results": json.loads(row["optimization_results"])
            }
            for row in rows
        ]
    
    # Pre-market Data Operations
    async def save_premarket_data(self, data: Dict[str, Any]):
        """Save pre-market data"""
        
        cursor = self.db_connection.cursor()
        
        # Extract date from data
        date_str = data.get("timestamp", datetime.now().isoformat()).split("T")[0]
        
        cursor.execute("""
            INSERT OR REPLACE INTO premarket_data (date, data)
            VALUES (?, ?)
        """, (date_str, json.dumps(data)))
        
        self.db_connection.commit()
    
    async def get_premarket_data(self, date: str = None) -> Optional[Dict[str, Any]]:
        """Get pre-market data"""
        
        cursor = self.db_connection.cursor()
        
        if date:
            cursor.execute("SELECT * FROM premarket_data WHERE date = ?", (date,))
        else:
            cursor.execute("SELECT * FROM premarket_data ORDER BY date DESC LIMIT 1")
        
        row = cursor.fetchone()
        
        if row:
            return json.loads(row["data"])
        return None
    
    # Metrics Operations
    async def save_agent_metric(self, agent_id: str, metric_name: str, metric_value: float):
        """Save agent metric"""
        
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO agent_metrics (agent_id, metric_name, metric_value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (agent_id, metric_name, metric_value, datetime.now()))
        
        self.db_connection.commit()
    
    async def get_agent_metrics(self, agent_id: str, metric_name: str = None, 
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get agent metrics"""
        
        cursor = self.db_connection.cursor()
        
        if metric_name:
            cursor.execute("""
                SELECT * FROM agent_metrics 
                WHERE agent_id = ? AND metric_name = ? 
                ORDER BY timestamp DESC LIMIT ?
            """, (agent_id, metric_name, limit))
        else:
            cursor.execute("""
                SELECT * FROM agent_metrics 
                WHERE agent_id = ? 
                ORDER BY timestamp DESC LIMIT ?
            """, (agent_id, limit))
        
        rows = cursor.fetchall()
        
        return [
            {
                "agent_id": row["agent_id"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "timestamp": row["timestamp"]
            }
            for row in rows
        ]
    
    # Configuration Operations
    async def save_agent_configuration(self, agent_id: str, configuration: Dict[str, Any]):
        """Save agent configuration"""
        
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO agent_configurations (agent_id, configuration, updated_at)
            VALUES (?, ?, ?)
        """, (agent_id, json.dumps(configuration), datetime.now()))
        
        self.db_connection.commit()
    
    async def get_agent_configuration(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration"""
        
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM agent_configurations WHERE agent_id = ?", (agent_id,))
        row = cursor.fetchone()
        
        if row:
            return json.loads(row["configuration"])
        return None
    
    # ChromaDB Operations
    async def _save_strategy_embedding(self, strategy: Dict[str, Any]):
        """Save strategy embedding to ChromaDB"""
        
        if not self.chroma_client or "strategy_embeddings" not in self.collections:
            return
        
        try:
            # Create embedding text from strategy
            embedding_text = f"""
            Strategy: {strategy.get('name', '')}
            Description: {strategy.get('description', '')}
            Components: {len(strategy.get('components', []))}
            Timeframes: {', '.join(strategy.get('timeframes', []))}
            Instruments: {', '.join(strategy.get('instruments', []))}
            """
            
            # Add to ChromaDB
            self.collections["strategy_embeddings"].add(
                documents=[embedding_text],
                ids=[strategy.get("strategy_id")],
                metadatas=[{
                    "strategy_id": strategy.get("strategy_id"),
                    "name": strategy.get("name", ""),
                    "created_at": strategy.get("created_at", "")
                }]
            )
            
        except Exception as e:
            log_error(f"Failed to save strategy embedding: {e}")
    
    async def search_similar_strategies(self, query: str, limit: int = 5) -> List[str]:
        """Search for similar strategies using ChromaDB"""
        
        if not self.chroma_client or "strategy_embeddings" not in self.collections:
            return []
        
        try:
            results = self.collections["strategy_embeddings"].query(
                query_texts=[query],
                n_results=limit
            )
            
            return results["ids"][0] if results["ids"] else []
            
        except Exception as e:
            log_error(f"Failed to search strategies: {e}")
            return []
    
    # Analytics Operations
    async def get_agent_performance_summary(self, agent_id: str, days: int = 30) -> Dict[str, Any]:
        """Get agent performance summary"""
        
        cursor = self.db_connection.cursor()
        
        # Get task statistics
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM agent_tasks 
            WHERE agent_id = ? AND created_at >= datetime('now', '-{} days')
            GROUP BY status
        """.format(days), (agent_id,))
        
        task_stats = {row["status"]: row["count"] for row in cursor.fetchall()}
        
        # Get metrics
        cursor.execute("""
            SELECT metric_name, AVG(metric_value) as avg_value 
            FROM agent_metrics 
            WHERE agent_id = ? AND timestamp >= datetime('now', '-{} days')
            GROUP BY metric_name
        """.format(days), (agent_id,))
        
        metrics = {row["metric_name"]: row["avg_value"] for row in cursor.fetchall()}
        
        return {
            "agent_id": agent_id,
            "period_days": days,
            "task_statistics": task_stats,
            "performance_metrics": metrics
        }
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview"""
        
        cursor = self.db_connection.cursor()
        
        # Agent count
        cursor.execute("SELECT COUNT(*) as count FROM agent_data")
        agent_count = cursor.fetchone()["count"]
        
        # Task count
        cursor.execute("SELECT status, COUNT(*) as count FROM agent_tasks GROUP BY status")
        task_stats = {row["status"]: row["count"] for row in cursor.fetchall()}
        
        # Strategy count
        cursor.execute("SELECT COUNT(*) as count FROM strategies")
        strategy_count = cursor.fetchone()["count"]
        
        # Message count
        cursor.execute("SELECT COUNT(*) as count FROM agent_communication")
        message_count = cursor.fetchone()["count"]
        
        return {
            "total_agents": agent_count,
            "task_statistics": task_stats,
            "total_strategies": strategy_count,
            "total_messages": message_count,
            "storage_type": "SQLite + ChromaDB" if self.chroma_client else "SQLite"
        }
    
    async def cleanup_old_data(self, days: int = 90):
        """Clean up old data"""
        
        cursor = self.db_connection.cursor()
        
        # Clean old tasks
        cursor.execute("""
            DELETE FROM agent_tasks 
            WHERE created_at < datetime('now', '-{} days')
        """.format(days))
        
        # Clean old messages
        cursor.execute("""
            DELETE FROM agent_communication 
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days))
        
        # Clean old metrics
        cursor.execute("""
            DELETE FROM agent_metrics 
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days))
        
        self.db_connection.commit()
        
        log_info(f"Cleaned up data older than {days} days")
    
    async def close(self):
        """Close database connections"""
        
        if self.db_connection:
            self.db_connection.close()
        
        if self.chroma_client:
            # ChromaDB doesn't need explicit closing
            pass
        
        log_info("Agent data store closed")
