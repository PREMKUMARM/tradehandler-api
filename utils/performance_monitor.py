"""
Performance monitoring and metrics collection
"""
import time
import asyncio
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from functools import wraps
from utils.logger import log_info, log_error, log_warning


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None


@dataclass
class RequestMetric:
    """HTTP request metric"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    user_id: Optional[str] = None


class PerformanceMonitor:
    """Advanced performance monitoring system"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.requests: List[RequestMetric] = []
        self.start_time = datetime.now()
        self.max_metrics = 10000
        self.max_requests = 10000
        self._lock = asyncio.Lock()
    
    async def record_metric(self, name: str, value: float, unit: str = "ms", tags: Dict[str, str] = None):
        """Record a performance metric"""
        async with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics.append(metric)
            
            # Keep only recent metrics
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]
    
    async def record_request(self, endpoint: str, method: str, status_code: int, 
                           response_time: float, user_id: Optional[str] = None):
        """Record HTTP request metric"""
        async with self._lock:
            request = RequestMetric(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time=response_time,
                timestamp=datetime.now(),
                user_id=user_id
            )
            self.requests.append(request)
            
            # Keep only recent requests
            if len(self.requests) > self.max_requests:
                self.requests = self.requests[-self.max_requests:]
    
    def get_system_metrics(self) -> Dict:
        """Get system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
        except Exception as e:
            log_error(f"Error getting system metrics: {e}")
            return {}
    
    async def get_metrics_summary(self, minutes: int = 5) -> Dict:
        """Get summary of recent metrics"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        async with self._lock:
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            recent_requests = [r for r in self.requests if r.timestamp > cutoff_time]
        
        # Calculate metric summaries
        metric_summary = {}
        for metric in recent_metrics:
            key = f"{metric.name}_{metric.unit}"
            if key not in metric_summary:
                metric_summary[key] = []
            metric_summary[key].append(metric.value)
        
        # Calculate statistics
        summary = {}
        for key, values in metric_summary.items():
            if values:
                summary[key] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1]
                }
        
        # Request statistics
        if recent_requests:
            response_times = [r.response_time for r in recent_requests]
            status_codes = {}
            for r in recent_requests:
                status_codes[r.status_code] = status_codes.get(r.status_code, 0) + 1
            
            summary['requests'] = {
                'total': len(recent_requests),
                'avg_response_time': sum(response_times) / len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'requests_per_minute': len(recent_requests) / minutes,
                'status_codes': status_codes,
                'success_rate': (len([r for r in recent_requests if 200 <= r.status_code < 300]) / len(recent_requests)) * 100
            }
        
        return summary
    
    async def get_endpoint_stats(self, endpoint: str = None, minutes: int = 60) -> Dict:
        """Get endpoint-specific statistics"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        async with self._lock:
            recent_requests = [r for r in self.requests if r.timestamp > cutoff_time]
            if endpoint:
                recent_requests = [r for r in recent_requests if r.endpoint == endpoint]
        
        if not recent_requests:
            return {}
        
        # Group by endpoint
        endpoints = {}
        for request in recent_requests:
            ep = request.endpoint
            if ep not in endpoints:
                endpoints[ep] = {
                    'requests': [],
                    'response_times': [],
                    'status_codes': {}
                }
            
            endpoints[ep]['requests'].append(request)
            endpoints[ep]['response_times'].append(request.response_time)
            
            status = request.status_code
            endpoints[ep]['status_codes'][status] = endpoints[ep]['status_codes'].get(status, 0) + 1
        
        # Calculate statistics for each endpoint
        stats = {}
        for ep, data in endpoints.items():
            response_times = data['response_times']
            stats[ep] = {
                'request_count': len(response_times),
                'avg_response_time': sum(response_times) / len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'success_rate': (len([r for r in data['requests'] if 200 <= r.status_code < 300]) / len(data['requests'])) * 100,
                'status_codes': data['status_codes'],
                'requests_per_minute': len(response_times) / minutes
            }
        
        return stats
    
    def get_uptime(self) -> Dict:
        """Get application uptime information"""
        uptime = datetime.now() - self.start_time
        
        return {
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': uptime.total_seconds(),
            'uptime_days': uptime.days,
            'uptime_hours': uptime.total_seconds() / 3600,
            'uptime_formatted': str(uptime).split('.')[0]  # Remove microseconds
        }
    
    async def clear_old_metrics(self, hours: int = 24):
        """Clear old metrics to prevent memory leaks"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        async with self._lock:
            old_metrics_count = len(self.metrics)
            old_requests_count = len(self.requests)
            
            self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            self.requests = [r for r in self.requests if r.timestamp > cutoff_time]
            
            cleared_metrics = old_metrics_count - len(self.metrics)
            cleared_requests = old_requests_count - len(self.requests)
            
            if cleared_metrics > 0 or cleared_requests > 0:
                log_info(f"Cleared {cleared_metrics} old metrics and {cleared_requests} old requests")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(metric_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            name = metric_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                # Record performance metric
                asyncio.create_task(
                    performance_monitor.record_metric(
                        name=name,
                        value=response_time,
                        unit="ms",
                        tags={"success": str(success)}
                    )
                )
        
        return wrapper
    return decorator


def monitor_requests():
    """Middleware to monitor HTTP requests"""
    async def middleware(request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            # Record request metric
            user_id = getattr(request.state, 'user_id', None)
            asyncio.create_task(
                performance_monitor.record_request(
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=status_code,
                    response_time=response_time,
                    user_id=user_id
                )
            )
        
        return response
    
    return middleware


# Background task to clear old metrics
async def start_metrics_cleanup():
    """Start background task to clean up old metrics"""
    while True:
        try:
            await performance_monitor.clear_old_metrics()
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            log_error(f"Error in metrics cleanup: {e}")
            await asyncio.sleep(300)  # Retry after 5 minutes
