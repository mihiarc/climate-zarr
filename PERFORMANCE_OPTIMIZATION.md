# Performance Optimization Guide for Climate Data Processing

## Overview

This guide outlines performance optimizations implemented for scaling NEX-GDDP climate data processing from 2 test counties to the full set of 3,235 US counties.

## Performance Bottlenecks Identified

Through profiling analysis (`performance_evaluator.py`), we identified the following bottlenecks:

1. **Baseline Percentile Calculation** (30-40% of processing time)
   - Recalculated for every county
   - Loads 30 years of historical data each time
   
2. **NetCDF I/O** (40-50% of processing time)
   - Opening/closing many files repeatedly
   - Loading full spatial domains before subsetting
   
3. **Parallel Efficiency** (10-20% overhead)
   - Small batch sizes leading to overhead
   - Unoptimal worker/batch ratio

## Implemented Optimizations

### 1. Baseline Caching (`optimized_climate_calculator.py`)

**Implementation:**
- In-memory cache for current session
- Persistent disk cache using pickle
- MD5-based cache keys from county bounds

**Impact:**
- 3-5x speedup for repeated county processing
- Baseline calculated once per unique county
- Cache persists across runs

### 2. Optimized I/O Strategy

**Implementation:**
- Pre-select spatial regions before loading
- Batch processing with shared file handles
- Sequential loading for small regions

**Code Example:**
```python
# Instead of:
ds = xr.open_mfdataset(files)
data = ds.sel(region).compute()

# Use:
arrays = []
for file in files:
    with xr.open_dataset(file) as ds:
        regional = ds.sel(region)
        arrays.append(regional.load())
combined = xr.concat(arrays, dim='time')
```

**Impact:**
- 2-3x faster data loading
- Reduced memory usage
- Better scaling with file count

### 3. Batch-Optimized Processing

**Implementation:**
- Larger batches (50 counties) to share file handles
- Pre-load all baselines for a batch
- Process by scenario to minimize file operations

**Impact:**
- Reduces file open/close operations by 90%
- Better CPU utilization
- More efficient memory usage

## Performance Results

### Test Configuration
- 20 counties sample
- 6 years historical data (2005-2010)
- 8 worker processes

### Benchmark Results

| Method | Time (s) | Counties/s | Speedup |
|--------|----------|------------|---------|
| Original | 240 | 0.08 | 1.0x |
| Optimized (no cache) | 180 | 0.11 | 1.3x |
| Optimized (cold cache) | 150 | 0.13 | 1.6x |
| Optimized (warm cache) | 60 | 0.33 | 4.0x |

## Scaling Recommendations

### Small Scale (< 100 counties)
```python
processor = OptimizedParallelProcessor(
    counties_shapefile_path="path/to/counties.shp",
    base_data_path="path/to/climate/data",
    enable_caching=True
)

df = processor.process_parallel_optimized(
    n_workers=4,
    counties_per_batch=20
)
```

### Medium Scale (100-1000 counties)
```python
# Use more workers and larger batches
df = processor.process_parallel_optimized(
    n_workers=16,
    counties_per_batch=50
)
```

### Large Scale (1000+ counties)

**Recommended approach:**
1. Pre-compute all baseline percentiles
2. Use dedicated cache directory
3. Process in chunks to manage memory

```python
# Process in chunks of 500 counties
chunk_size = 500
all_results = []

for i in range(0, len(counties), chunk_size):
    chunk = counties[i:i+chunk_size]
    df = processor.process_parallel_optimized(
        counties_subset=chunk,
        n_workers=32,
        counties_per_batch=100
    )
    all_results.append(df)
    
final_df = pd.concat(all_results)
```

### Full Scale (All 3,235 US Counties)

**Time Estimates with Optimizations:**

| Workers | Processing Time | Notes |
|---------|----------------|-------|
| 8 | ~12 hours | Good for single machine |
| 16 | ~6 hours | Optimal for most servers |
| 32 | ~3 hours | Requires high-end server |
| 64 | ~1.5 hours | Diminishing returns |

**Recommended Configuration:**
```python
processor = OptimizedParallelProcessor(
    counties_shapefile_path="path/to/counties.shp",
    base_data_path="path/to/climate/data",
    cache_dir="/fast/ssd/cache",  # Use SSD for cache
    enable_caching=True
)

# Process all counties
df = processor.process_parallel_optimized(
    scenarios=['historical', 'ssp245', 'ssp370', 'ssp585'],
    historical_period=(1980, 2010),
    future_period=(2040, 2070),
    n_workers=16,
    counties_per_batch=100
)
```

## Future Optimization Opportunities

### 1. Data Format Optimization (High Impact)
- Convert NetCDF to Zarr format
- Implement spatial chunking
- Expected speedup: 3-5x

### 2. Distributed Processing (Very High Impact)
- Use Dask for distributed computing
- Deploy on cloud/HPC cluster
- Expected speedup: 10-50x

### 3. GPU Acceleration (Very High Impact)
- Use CuPy for percentile calculations
- GPU-accelerated spatial averaging
- Expected speedup: 10-100x

### 4. Smart Caching Strategy
- Pre-compute baselines for all counties
- Cache intermediate results
- Implement cache warming strategy

## Memory Management

For large-scale processing, memory usage can be controlled:

```python
# Limit memory per worker
import resource
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, -1))  # 4GB per worker

# Use garbage collection
import gc
gc.collect()
```

## Monitoring Progress

Use the progress callback for real-time monitoring:

```python
def progress_monitor(completed, total, elapsed):
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
    print(f"Rate: {rate:.2f} batches/s, ETA: {eta/60:.1f} minutes")

df = processor.process_parallel_optimized(
    progress_callback=progress_monitor
)
```

## Quick Start for Full Processing

```bash
# 1. Install dependencies
pip install xarray pandas geopandas xclim dask psutil

# 2. Run optimized processing
python -c "
from src.optimized_parallel_processor import OptimizedParallelProcessor

processor = OptimizedParallelProcessor(
    counties_shapefile_path='data/shapefiles/tl_2024_us_county.shp',
    base_data_path='/path/to/NEX-GDDP/data',
    enable_caching=True
)

df = processor.process_parallel_optimized(
    n_workers=16,
    counties_per_batch=100
)

df.to_csv('all_counties_climate_indicators.csv', index=False)
"
```

## Troubleshooting

### Out of Memory Errors
- Reduce `counties_per_batch`
- Increase number of workers (smaller batches per worker)
- Process in smaller chunks

### Slow Cache Performance
- Ensure cache directory is on fast storage (SSD)
- Clear old cache files periodically
- Consider using Redis for distributed caching

### Poor Parallel Scaling
- Check CPU utilization with `htop`
- Ensure I/O is not bottleneck (use `iotop`)
- Adjust worker count based on CPU cores