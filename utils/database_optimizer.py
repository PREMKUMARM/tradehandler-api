"""
Database optimization utilities including connection pooling and query optimization
"""
import asyncio
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import sqlite3
import aiosqlite
from datetime import datetime, timedelta
from utils.logger import log_info, log_error, log_warning


class DatabasePool:
    """Async SQLite connection pool"""
    
    def __init__(self, database_path: str, max_connections: int = 10):
        self.database_path = database_path
        self.max_connections = max_connections
        self._pool = asyncio.Queue(maxsize=max_connections)
        self._created_connections = 0
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the connection pool"""
        # Create initial connections
        for _ in range(min(3, self.max_connections)):
            await self._create_connection()
    
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection"""
        async with self._lock:
            if self._created_connections >= self.max_connections:
                raise Exception("Maximum connections reached")
            
            conn = await aiosqlite.connect(self.database_path)
            
            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            
            # Optimize for performance
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=10000")
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            
            # Enable foreign keys
            await conn.execute("PRAGMA foreign_keys=ON")
            
            self._created_connections += 1
            return conn
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        try:
            # Try to get existing connection
            conn = self._pool.get_nowait()
        except asyncio.QueueEmpty:
            # Create new connection if pool is empty
            conn = await self._create_connection()
        
        try:
            yield conn
        except Exception:
            # Rollback on error
            await conn.rollback()
            raise
        finally:
            # Return connection to pool
            try:
                self._pool.put_nowait(conn)
            except asyncio.QueueFull:
                # Pool is full, close the connection
                await conn.close()
                async with self._lock:
                    self._created_connections -= 1
    
    async def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
        
        self._created_connections = 0


class QueryOptimizer:
    """Database query optimization utilities"""
    
    def __init__(self, db_pool: DatabasePool):
        self.db_pool = db_pool
        self.query_stats: Dict[str, Dict] = {}
    
    async def execute_query(self, query: str, params: tuple = None, fetch: str = "all") -> Any:
        """Execute query with performance tracking"""
        start_time = time.time()
        
        async with self.db_pool.get_connection() as conn:
            conn.row_factory = aiosqlite.Row
            
            try:
                cursor = await conn.execute(query, params or ())
                
                if fetch == "all":
                    result = await cursor.fetchall()
                elif fetch == "one":
                    result = await cursor.fetchone()
                elif fetch == "many":
                    result = await cursor.fetchmany()
                else:
                    result = None
                
                await conn.commit()
                
                # Track query performance
                execution_time = time.time() - start_time
                self._track_query(query, execution_time, len(result) if isinstance(result, list) else 1)
                
                return result
                
            except Exception as e:
                await conn.rollback()
                log_error(f"Query execution error: {e}")
                raise
    
    async def execute_many(self, query: str, params_list: List[tuple]) -> None:
        """Execute query with multiple parameter sets"""
        start_time = time.time()
        
        async with self.db_pool.get_connection() as conn:
            try:
                await conn.executemany(query, params_list)
                await conn.commit()
                
                # Track performance
                execution_time = time.time() - start_time
                self._track_query(query, execution_time, len(params_list))
                
            except Exception as e:
                await conn.rollback()
                log_error(f"Batch execution error: {e}")
                raise
    
    def _track_query(self, query: str, execution_time: float, rows_affected: int):
        """Track query performance statistics"""
        query_hash = hash(query)
        
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = {
                "query": query,
                "count": 0,
                "total_time": 0,
                "avg_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "total_rows": 0
            }
        
        stats = self.query_stats[query_hash]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["total_rows"] += rows_affected
        
        # Log slow queries
        if execution_time > 1.0:  # Queries taking more than 1 second
            log_warning(f"Slow query detected: {execution_time:.2f}s - {query[:100]}...")
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        return {
            "total_queries": len(self.query_stats),
            "slow_queries": len([s for s in self.query_stats.values() if s["avg_time"] > 1.0]),
            "queries": list(self.query_stats.values())
        }
    
    async def optimize_database(self):
        """Run database optimization commands"""
        optimization_queries = [
            "ANALYZE",
            "VACUUM",
            "REINDEX"
        ]
        
        async with self.db_pool.get_connection() as conn:
            for query in optimization_queries:
                try:
                    log_info(f"Running database optimization: {query}")
                    await conn.execute(query)
                    await conn.commit()
                except Exception as e:
                    log_error(f"Database optimization error ({query}): {e}")


class CacheManager:
    """Database-backed cache manager"""
    
    def __init__(self, db_pool: DatabasePool):
        self.db_pool = db_pool
        self.default_ttl = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize cache tables"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            expires_at REAL NOT NULL,
            created_at REAL NOT NULL
        )
        """
        
        async with self.db_pool.get_connection() as conn:
            await conn.execute(create_table_query)
            await conn.commit()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        query = "SELECT value FROM cache WHERE key = ? AND expires_at > ?"
        
        async with self.db_pool.get_connection() as conn:
            cursor = await conn.execute(query, (key, time.time()))
            row = await cursor.fetchone()
            
            if row:
                import json
                return json.loads(row[0])
            
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value"""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        import json
        value_json = json.dumps(value, default=str)
        
        query = """
        INSERT OR REPLACE INTO cache (key, value, expires_at, created_at)
        VALUES (?, ?, ?, ?)
        """
        
        async with self.db_pool.get_connection() as conn:
            await conn.execute(query, (key, value_json, expires_at, time.time()))
            await conn.commit()
    
    async def delete(self, key: str) -> bool:
        """Delete cached value"""
        query = "DELETE FROM cache WHERE key = ?"
        
        async with self.db_pool.get_connection() as conn:
            cursor = await conn.execute(query, (key,))
            await conn.commit()
            return cursor.rowcount > 0
    
    async def clear_expired(self) -> int:
        """Clear expired cache entries"""
        query = "DELETE FROM cache WHERE expires_at <= ?"
        
        async with self.db_pool.get_connection() as conn:
            cursor = await conn.execute(query, (time.time(),))
            await conn.commit()
            return cursor.rowcount
    
    async def clear_all(self) -> int:
        """Clear all cache entries"""
        query = "DELETE FROM cache"
        
        async with self.db_pool.get_connection() as conn:
            cursor = await conn.execute(query)
            await conn.commit()
            return cursor.rowcount


# Global instances
db_pool: Optional[DatabasePool] = None
query_optimizer: Optional[QueryOptimizer] = None
cache_manager: Optional[CacheManager] = None


async def initialize_database(database_path: str = "data/algofeast.db"):
    """Initialize database with optimizations"""
    global db_pool, query_optimizer, cache_manager
    
    # Initialize connection pool
    db_pool = DatabasePool(database_path)
    await db_pool.initialize()
    
    # Initialize query optimizer
    query_optimizer = QueryOptimizer(db_pool)
    
    # Initialize cache manager
    cache_manager = CacheManager(db_pool)
    await cache_manager.initialize()
    
    log_info("Database initialized with optimizations")


async def get_db_connection():
    """Get database connection from pool"""
    if not db_pool:
        raise Exception("Database not initialized")
    
    return db_pool.get_connection()


def cached_query(ttl: int = 300):
    """Decorator for caching query results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            if cache_manager:
                cached_result = await cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            if cache_manager:
                await cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


async def cleanup_expired_cache():
    """Background task to clean up expired cache entries"""
    while True:
        try:
            if cache_manager:
                cleared = await cache_manager.clear_expired()
                if cleared > 0:
                    log_info(f"Cleared {cleared} expired cache entries")
            
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            log_error(f"Cache cleanup error: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute
