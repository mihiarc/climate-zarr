#!/usr/bin/env python
"""Memory management utilities for optimal chunked processing."""

import gc
import psutil
import numpy as np
from typing import Optional, Tuple, Dict
from rich.console import Console

console = Console()


class MemoryMonitor:
    """Real-time memory monitoring for chunked processing."""
    
    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0):
        """Initialize memory monitor.
        
        Args:
            warning_threshold: Memory percentage to trigger warnings
            critical_threshold: Memory percentage to trigger critical alerts
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.initial_memory = psutil.virtual_memory().percent
        
    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory status."""
        memory = psutil.virtual_memory()
        return {
            'percent_used': memory.percent,
            'available_gb': memory.available / (1024**3),
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3)
        }
    
    def check_memory_pressure(self) -> str:
        """Check memory pressure level.
        
        Returns:
            'normal', 'warning', or 'critical'
        """
        current_percent = psutil.virtual_memory().percent
        
        if current_percent >= self.critical_threshold:
            return 'critical'
        elif current_percent >= self.warning_threshold:
            return 'warning'
        else:
            return 'normal'
    
    def should_reduce_chunk_size(self) -> bool:
        """Determine if chunk size should be reduced due to memory pressure."""
        return self.check_memory_pressure() in ['warning', 'critical']
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup."""
        gc.collect()
        
        # Additional cleanup for scientific computing
        import numpy as np
        try:
            # Clear numpy's temporary array cache if it exists
            if hasattr(np, '_NoValue'):
                np._NoValue = None
        except:
            pass


def estimate_chunk_memory_usage(
    data_shape: Tuple[int, ...],
    dtype_size: int,
    counties_in_chunk: int,
    boundary_complexity_factor: float = 1.5
) -> float:
    """Estimate memory usage for a chunk of counties.
    
    Args:
        data_shape: Shape of the climate data array (time, lat, lon)
        dtype_size: Size of data type in bytes (e.g., 8 for float64)
        counties_in_chunk: Number of counties in the chunk
        boundary_complexity_factor: Factor for boundary complexity (1.0-3.0)
        
    Returns:
        Estimated memory usage in GB
    """
    if len(data_shape) < 3:
        raise ValueError("Data shape must have at least 3 dimensions (time, lat, lon)")
    
    time_steps, lat_size, lon_size = data_shape[0], data_shape[1], data_shape[2]
    
    # Base memory for the full dataset
    base_memory_gb = (time_steps * lat_size * lon_size * dtype_size) / (1024**3)
    
    # Estimate memory per county (assuming uniform distribution)
    avg_county_memory_gb = base_memory_gb / 3109  # Approximate CONUS county count
    
    # Account for:
    # 1. Multiple counties in memory simultaneously
    # 2. Boundary complexity (coastal counties, complex shapes)
    # 3. Intermediate processing arrays
    # 4. Python object overhead
    processing_overhead = 2.0  # 2x overhead for intermediate processing
    
    estimated_gb = (avg_county_memory_gb * counties_in_chunk * 
                   boundary_complexity_factor * processing_overhead)
    
    return estimated_gb


def calculate_optimal_chunk_size(
    data_shape: Tuple[int, ...],
    available_memory_gb: float,
    target_memory_usage: float = 0.75,
    min_chunk_size: int = 5,
    max_chunk_size: int = 50
) -> int:
    """Calculate optimal chunk size based on available memory.
    
    Args:
        data_shape: Shape of the climate data
        available_memory_gb: Available system memory in GB
        target_memory_usage: Target memory usage ratio (0.0-1.0)
        min_chunk_size: Minimum counties per chunk
        max_chunk_size: Maximum counties per chunk
        
    Returns:
        Optimal chunk size (number of counties)
    """
    target_memory_gb = available_memory_gb * target_memory_usage
    
    # Binary search for optimal chunk size
    low, high = min_chunk_size, max_chunk_size
    optimal_size = min_chunk_size
    
    while low <= high:
        mid = (low + high) // 2
        estimated_memory = estimate_chunk_memory_usage(
            data_shape, 8, mid  # Assuming float64
        )
        
        if estimated_memory <= target_memory_gb:
            optimal_size = mid
            low = mid + 1
        else:
            high = mid - 1
    
    console.print(f"[cyan]Optimal chunk size: {optimal_size} counties[/cyan]")
    console.print(f"[cyan]Estimated memory per chunk: {estimate_chunk_memory_usage(data_shape, 8, optimal_size):.2f} GB[/cyan]")
    
    return optimal_size


def adaptive_chunk_sizing(
    base_chunk_size: int,
    memory_monitor: MemoryMonitor,
    processing_time_per_chunk: Optional[float] = None
) -> int:
    """Adaptively adjust chunk size based on current system conditions.
    
    Args:
        base_chunk_size: Base chunk size to adjust
        memory_monitor: Memory monitor instance
        processing_time_per_chunk: Recent processing time per chunk (seconds)
        
    Returns:
        Adjusted chunk size
    """
    memory_status = memory_monitor.check_memory_pressure()
    
    if memory_status == 'critical':
        # Reduce chunk size significantly
        adjusted = max(1, base_chunk_size // 4)
        console.print(f"[red]Critical memory pressure: reducing chunk size to {adjusted}[/red]")
        return adjusted
    
    elif memory_status == 'warning':
        # Reduce chunk size moderately
        adjusted = max(1, base_chunk_size // 2)
        console.print(f"[yellow]Memory warning: reducing chunk size to {adjusted}[/yellow]")
        return adjusted
    
    else:
        # Normal memory conditions - consider increasing if processing is fast
        if processing_time_per_chunk and processing_time_per_chunk < 30:  # Less than 30 seconds
            adjusted = min(50, int(base_chunk_size * 1.2))
            if adjusted > base_chunk_size:
                console.print(f"[green]Fast processing: increasing chunk size to {adjusted}[/green]")
            return adjusted
    
    return base_chunk_size


class ChunkPerformanceTracker:
    """Track chunk processing performance for optimization."""
    
    def __init__(self):
        self.chunk_times = []
        self.chunk_sizes = []
        self.memory_usage = []
    
    def record_chunk_performance(self, chunk_size: int, processing_time: float, 
                                peak_memory_percent: float):
        """Record performance metrics for a chunk."""
        self.chunk_sizes.append(chunk_size)
        self.chunk_times.append(processing_time)
        self.memory_usage.append(peak_memory_percent)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.chunk_times:
            return {}
        
        return {
            'avg_time_per_county': np.mean([t/s for t, s in zip(self.chunk_times, self.chunk_sizes)]),
            'avg_chunk_time': np.mean(self.chunk_times),
            'avg_memory_usage': np.mean(self.memory_usage),
            'total_chunks': len(self.chunk_times),
            'fastest_time_per_county': min([t/s for t, s in zip(self.chunk_times, self.chunk_sizes)]),
            'peak_memory_usage': max(self.memory_usage) if self.memory_usage else 0.0
        }
    
    def recommend_optimal_chunk_size(self) -> Optional[int]:
        """Recommend optimal chunk size based on historical performance."""
        if len(self.chunk_times) < 3:
            return None
        
        # Find chunk size with best time-per-county ratio under acceptable memory usage
        time_per_county = [t/s for t, s in zip(self.chunk_times, self.chunk_sizes)]
        
        # Filter out chunks that used excessive memory
        acceptable_indices = [i for i, mem in enumerate(self.memory_usage) if mem < 85.0]
        
        if not acceptable_indices:
            return min(self.chunk_sizes)  # Conservative fallback
        
        best_idx = min(acceptable_indices, key=lambda i: time_per_county[i])
        return self.chunk_sizes[best_idx]