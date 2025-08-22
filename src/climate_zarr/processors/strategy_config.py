#!/usr/bin/env python
"""Configuration and selection logic for processing strategies."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import psutil
import xarray as xr
import geopandas as gpd
from rich.console import Console

from .processing_strategies import VectorizedStrategy, SpatialChunkedStrategy

console = Console()


@dataclass
class ProcessingConfig:
    """Configuration for processing strategy selection."""
    
    # Memory configuration
    target_memory_usage: float = 0.75  # Use 75% of available memory
    min_chunk_size: int = 5
    max_chunk_size: int = 50
    
    # Performance configuration
    enable_spatial_cache: bool = True
    enable_adaptive_chunking: bool = True
    parallel_workers: int = 4
    
    # Strategy selection thresholds
    large_dataset_threshold_gb: float = 10.0  # Switch to chunked strategy
    many_counties_threshold: int = 500        # Switch to chunked strategy
    low_memory_threshold_gb: float = 8.0      # Use conservative settings
    
    @classmethod
    def create_optimized_config(cls, 
                               data: xr.DataArray, 
                               gdf: gpd.GeoDataFrame,
                               available_memory_gb: Optional[float] = None) -> 'ProcessingConfig':
        """Create optimized configuration based on data characteristics."""
        
        if available_memory_gb is None:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Calculate dataset size
        dataset_size_gb = data.nbytes / (1024**3)
        num_counties = len(gdf)
        
        console.print(f"[cyan]Dataset size: {dataset_size_gb:.2f} GB[/cyan]")
        console.print(f"[cyan]Number of counties: {num_counties}[/cyan]")
        console.print(f"[cyan]Available memory: {available_memory_gb:.2f} GB[/cyan]")
        
        config = cls()
        
        # Adjust memory usage based on available memory
        if available_memory_gb < config.low_memory_threshold_gb:
            config.target_memory_usage = 0.6  # Conservative memory usage
            config.max_chunk_size = 20
            console.print("[yellow]Low memory system detected - using conservative settings[/yellow]")
        
        # Adjust chunk sizes based on dataset characteristics
        if dataset_size_gb > config.large_dataset_threshold_gb:
            # Large dataset - use smaller chunks to avoid memory issues
            config.min_chunk_size = max(3, config.min_chunk_size // 2)
            config.max_chunk_size = min(30, config.max_chunk_size)
            console.print("[yellow]Large dataset detected - reducing chunk sizes[/yellow]")
        
        # Adjust parallelism based on system resources
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        config.parallel_workers = min(config.parallel_workers, cpu_count, 
                                     max(1, int(available_memory_gb // 4)))
        
        console.print(f"[green]Optimized configuration: {config.parallel_workers} workers, "
                     f"chunks {config.min_chunk_size}-{config.max_chunk_size}, "
                     f"{config.target_memory_usage*100:.0f}% memory target[/green]")
        
        return config


class StrategySelector:
    """Intelligent strategy selection based on data characteristics."""
    
    @staticmethod
    def select_optimal_strategy(
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        config: Optional[ProcessingConfig] = None
    ) -> tuple[Any, ProcessingConfig]:
        """Select the optimal processing strategy.
        
        Args:
            data: Climate data array
            gdf: County geometries
            config: Processing configuration (auto-generated if None)
            
        Returns:
            Tuple of (strategy_instance, final_config)
        """
        
        if config is None:
            config = ProcessingConfig.create_optimized_config(data, gdf)
        
        # Calculate decision metrics
        dataset_size_gb = data.nbytes / (1024**3)
        num_counties = len(gdf)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Decision logic for strategy selection
        use_chunked_strategy = (
            dataset_size_gb > config.large_dataset_threshold_gb or
            num_counties > config.many_counties_threshold or
            available_memory_gb > 16.0  # Plenty of memory for parallel processing
        )
        
        if use_chunked_strategy:
            console.print("[green]Selected: SpatialChunkedStrategy (optimal for large datasets and parallel processing)[/green]")
            strategy = SpatialChunkedStrategy(
                target_memory_usage=config.target_memory_usage,
                min_chunk_size=config.min_chunk_size,
                max_chunk_size=config.max_chunk_size,
                enable_spatial_cache=config.enable_spatial_cache
            )
            
            # Log strategy benefits
            console.print("[cyan]Strategy benefits:[/cyan]")
            console.print(f"  • Parallel processing with {config.parallel_workers} workers")
            console.print(f"  • Target memory usage: {config.target_memory_usage*100:.0f}%")
            console.print("  • Spatial locality optimization for cache efficiency")
            console.print("  • Adaptive chunk sizing based on county complexity")
            console.print("  • Graceful error handling with fallback processing")
            
        else:
            console.print("[green]Selected: VectorizedStrategy (optimal for smaller datasets and memory-constrained systems)[/green]")
            strategy = VectorizedStrategy()
            
            # Log strategy benefits
            console.print("[cyan]Strategy benefits:[/cyan]")
            console.print("  • Sequential processing with predictable memory usage")
            console.print("  • Precise geometric clipping for all county types")
            console.print("  • Optimized for coastal counties and complex boundaries")
            console.print("  • Lower memory overhead and simpler error handling")
        
        return strategy, config
    
    @staticmethod
    def estimate_processing_time(
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        strategy_type: str,
        config: ProcessingConfig
    ) -> Dict[str, float]:
        """Estimate processing time for different strategies.
        
        Args:
            data: Climate data array
            gdf: County geometries  
            strategy_type: 'vectorized' or 'chunked'
            config: Processing configuration
            
        Returns:
            Dictionary with time estimates
        """
        
        # Base processing rate estimates (counties per minute)
        # These are rough estimates based on typical performance
        base_rates = {
            'vectorized': 6.0,  # ~6 counties per minute
            'chunked': 20.0     # ~20 counties per minute with parallelism
        }
        
        num_counties = len(gdf)
        dataset_size_gb = data.nbytes / (1024**3)
        
        # Adjust for dataset complexity
        complexity_factor = min(2.0, 1.0 + dataset_size_gb / 20.0)  # Larger datasets take longer per county
        
        # Adjust for parallelism (chunked strategy only)
        if strategy_type == 'chunked':
            parallel_speedup = min(config.parallel_workers * 0.7, 4.0)  # Diminishing returns
        else:
            parallel_speedup = 1.0
        
        # Calculate time estimates
        base_rate = base_rates[strategy_type]
        adjusted_rate = base_rate * parallel_speedup / complexity_factor
        
        estimated_minutes = num_counties / adjusted_rate
        
        return {
            'estimated_minutes': estimated_minutes,
            'estimated_hours': estimated_minutes / 60.0,
            'counties_per_minute': adjusted_rate,
            'parallel_speedup': parallel_speedup,
            'complexity_factor': complexity_factor
        }


def create_processing_plan(
    data: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    variable: str,
    scenario: str = 'historical',
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a comprehensive processing plan.
    
    Args:
        data: Climate data array
        gdf: County geometries
        variable: Climate variable name
        scenario: Scenario name
        custom_config: Custom configuration overrides
        
    Returns:
        Complete processing plan with strategy, config, and estimates
    """
    
    # Create base configuration
    config = ProcessingConfig.create_optimized_config(data, gdf)
    
    # Apply custom overrides
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
                console.print(f"[yellow]Config override: {key} = {value}[/yellow]")
    
    # Select strategy
    strategy, final_config = StrategySelector.select_optimal_strategy(data, gdf, config)
    strategy_type = 'chunked' if isinstance(strategy, SpatialChunkedStrategy) else 'vectorized'
    
    # Generate time estimates
    time_estimates = StrategySelector.estimate_processing_time(data, gdf, strategy_type, final_config)
    
    # Create processing plan
    plan = {
        'strategy': strategy,
        'strategy_type': strategy_type,
        'config': final_config,
        'data_characteristics': {
            'dataset_size_gb': data.nbytes / (1024**3),
            'num_counties': len(gdf),
            'time_steps': data.sizes.get('time', 1),
            'spatial_resolution': {
                'lat': abs(float(data.y[1] - data.y[0])) if len(data.y) > 1 else 0,
                'lon': abs(float(data.x[1] - data.x[0])) if len(data.x) > 1 else 0
            }
        },
        'system_resources': {
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_cores': psutil.cpu_count(logical=False),
            'target_memory_usage_gb': psutil.virtual_memory().available / (1024**3) * final_config.target_memory_usage
        },
        'time_estimates': time_estimates,
        'processing_parameters': {
            'variable': variable,
            'scenario': scenario,
            'workers': final_config.parallel_workers,
            'chunk_size_range': (final_config.min_chunk_size, final_config.max_chunk_size)
        }
    }
    
    # Display plan summary
    console.print("\n[bold blue]Processing Plan Summary[/bold blue]")
    console.print(f"Strategy: {strategy_type.title()}Strategy")
    console.print(f"Dataset: {plan['data_characteristics']['dataset_size_gb']:.2f} GB, "
                 f"{plan['data_characteristics']['num_counties']} counties")
    console.print(f"Estimated time: {time_estimates['estimated_hours']:.1f} hours "
                 f"({time_estimates['estimated_minutes']:.0f} minutes)")
    console.print(f"Processing rate: {time_estimates['counties_per_minute']:.1f} counties/minute")
    console.print(f"Memory target: {plan['system_resources']['target_memory_usage_gb']:.1f} GB\n")
    
    return plan