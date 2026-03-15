"""
Advanced caching manager for performance optimization
"""
import json
import time
import asyncio
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
import hashlib
from functools import wraps
from utils.logger import log_info, log_error, log_warning


class CacheManager:
    """Advanced caching system with TTL and memory management"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key not in self.cache:
                return None
            
            item = self.cache[key]
            current_time = time.time()
            
            # Check if expired
            if current_time > item['expires_at']:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return None
            
            # Update access time for LRU
            self.access_times[key] = current_time
            return item['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        async with self._lock:
            current_time = time.time()
            ttl = ttl or self.default_ttl
            
            # Remove oldest items if cache is full
            if len(self.cache) >= self.max_size:
                await self._evict_lru()
            
            self.cache[key] = {
                'value': value,
                'created_at': current_time,
                'expires_at': current_time + ttl,
                'ttl': ttl
            }
            self.access_times[key] = current_time
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache"""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache"""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
    
    async def _evict_lru(self) -> None:
        """Evict least recently used items"""
        if not self.access_times:
            return
        
        # Find oldest access time
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove oldest item
        if oldest_key in self.cache:
            del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'memory_usage': sum(len(str(item['value'])) for item in self.cache.values()),
            'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_count', 1), 1)
        }


# Global cache instance
cache_manager = CacheManager()


def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_manager._generate_key(
                f"{key_prefix}{func.__name__}", args, kwargs
            )
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        
        # Add cache management methods to wrapper
        wrapper.cache_clear = lambda: asyncio.create_task(cache_manager.clear())
        wrapper.cache_delete = lambda key: asyncio.create_task(cache_manager.delete(key))
        wrapper.cache_stats = lambda: cache_manager.get_stats()
        
        return wrapper
    return decorator


class RedisCacheManager:
    """Redis-based cache manager for distributed caching"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 300):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._redis = None
    
    async def _get_redis(self):
        """Get Redis connection"""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url)
            except ImportError:
                log_warning("Redis not available, falling back to memory cache")
                return None
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        redis_client = await self._get_redis()
        if not redis_client:
            return await cache_manager.get(key)
        
        try:
            value = await redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            log_error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache"""
        redis_client = await self._get_redis()
        if not redis_client:
            await cache_manager.set(key, value, ttl)
            return
        
        try:
            ttl = ttl or self.default_ttl
            await redis_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            log_error(f"Redis set error: {e}")
            await cache_manager.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete item from Redis cache"""
        redis_client = await self._get_redis()
        if not redis_client:
            return await cache_manager.delete(key)
        
        try:
            result = await redis_client.delete(key)
            return result > 0
        except Exception as e:
            log_error(f"Redis delete error: {e}")
            return await cache_manager.delete(key)
    
    async def clear(self) -> None:
        """Clear all cache"""
        redis_client = await self._get_redis()
        if not redis_client:
            await cache_manager.clear()
            return
        
        try:
            await redis_client.flushdb()
        except Exception as e:
            log_error(f"Redis clear error: {e}")
            await cache_manager.clear()


# Try to use Redis if available, otherwise fall back to memory cache
try:
    redis_cache = RedisCacheManager()
except Exception:
    redis_cache = None
