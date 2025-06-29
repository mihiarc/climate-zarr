# Performance Optimization Recommendations

Based on performance test results showing baseline calculation takes 245s per county (80% of processing time), here are targeted recommendations:

## Immediate Optimizations (Days)

### 1. Pre-compute All County Baselines
**Impact: Eliminate 245s per county**

```python
# Script to pre-compute all baselines
from src.optimized_climate_calculator import OptimizedClimateCalculator
from src.parallel_processor import ParallelClimateProcessor

def precompute_all_baselines():
    """Pre-compute baselines for all US counties."""
    processor = ParallelClimateProcessor(
        counties_shapefile_path="data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/path/to/climate/data"
    )
    
    calculator = OptimizedClimateCalculator(
        base_data_path="/path/to/climate/data",
        enable_caching=True
    )
    
    for idx, county in processor.counties.iterrows():
        county_info = processor.prepare_county_info(county)
        print(f"Computing baseline for {county_info['name']}...")
        calculator.calculate_baseline_percentiles(county_info['bounds'])
    
    print(f"Pre-computed baselines for {len(processor.counties)} counties")
```

**Expected Result**: 
- One-time computation: ~12 hours
- Future runs: 1.5 hours for all counties

### 2. Optimize Baseline Data Loading
**Current issue**: Loading 30 years × 365 days = 10,950 files per variable

```python
# Option A: Pre-merge baseline years
# Merge 1980-2010 into single files per variable
cdo mergetime tasmax_*_historical_*_{1980..2010}.nc tasmax_historical_baseline.nc

# Option B: Use Zarr format for baseline data
# Convert baseline period to Zarr for faster access
ds = xr.open_mfdataset(baseline_files)
ds.to_zarr('baseline_data.zarr')
```

**Expected speedup**: 5-10x for baseline calculation

### 3. Spatial Indexing for County Data
**Create county-specific data subsets**

```python
# Pre-extract county regions from climate data
def create_county_subsets(county_id, bounds):
    """Extract and save county-specific climate data."""
    for var in ['tas', 'tasmax', 'tasmin', 'pr']:
        for year in range(1980, 2011):
            # Load full file
            ds = xr.open_dataset(f"{var}_historical_{year}.nc")
            # Extract county region
            county_data = ds.sel(
                lat=slice(bounds[1]-0.5, bounds[3]+0.5),
                lon=slice(bounds[0]-0.5, bounds[2]+0.5)
            )
            # Save county subset
            county_data.to_netcdf(f"county_data/{county_id}/{var}_{year}.nc")
```

**Expected speedup**: 3-5x for data loading

## Medium-term Optimizations (Weeks)

### 4. Distributed Cache Service
**Use Redis/Memcached for shared cache across workers**

```python
import redis
import pickle

class DistributedCache:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379)
    
    def get_baseline(self, county_id):
        data = self.redis.get(f"baseline_{county_id}")
        return pickle.loads(data) if data else None
    
    def set_baseline(self, county_id, baseline):
        self.redis.set(f"baseline_{county_id}", pickle.dumps(baseline))
```

**Benefits**:
- Share cache across multiple machines
- Persistent cache between runs
- Better memory management

### 5. GPU Acceleration for Percentiles
**Use CuPy for percentile calculations**

```python
import cupy as cp

def calculate_percentiles_gpu(data):
    """Calculate percentiles on GPU."""
    # Transfer to GPU
    gpu_data = cp.asarray(data)
    
    # Group by day of year
    doy_groups = {}
    for i, date in enumerate(dates):
        doy = date.dayofyear
        if doy not in doy_groups:
            doy_groups[doy] = []
        doy_groups[doy].append(gpu_data[i])
    
    # Calculate percentiles on GPU
    percentiles = {}
    for doy, values in doy_groups.items():
        gpu_values = cp.array(values)
        percentiles[doy] = cp.percentile(gpu_values, [10, 90])
    
    return percentiles
```

**Expected speedup**: 10-50x for percentile calculation

## Long-term Optimizations (Months)

### 6. Cloud-Native Architecture
**Redesign for cloud storage and computing**

- Store data in cloud-optimized formats (Zarr, COG)
- Use Dask for distributed computing
- Deploy on cloud compute clusters

```python
# Dask distributed processing
from dask.distributed import Client
import xarray as xr

client = Client('scheduler-address:8786')

# Open cloud-optimized dataset
ds = xr.open_zarr('s3://bucket/climate-data.zarr')

# Process in parallel across cluster
results = []
for county in counties:
    future = client.submit(process_county, county, ds)
    results.append(future)

# Gather results
final_results = client.gather(results)
```

### 7. Incremental Processing
**Only process new/changed data**

```python
class IncrementalProcessor:
    def __init__(self):
        self.processed_tracker = self.load_processed_list()
    
    def needs_processing(self, county_id, scenario, year):
        key = f"{county_id}_{scenario}_{year}"
        return key not in self.processed_tracker
    
    def mark_processed(self, county_id, scenario, year):
        key = f"{county_id}_{scenario}_{year}"
        self.processed_tracker.add(key)
        self.save_processed_list()
```

## Recommended Implementation Order

1. **Week 1**: Pre-compute all baselines (immediate 8x speedup)
2. **Week 2**: Implement distributed cache (better scaling)
3. **Week 3**: Optimize baseline data format (faster baseline computation)
4. **Month 2**: GPU acceleration (if available)
5. **Month 3**: Cloud architecture (for production scale)

## Expected Performance After Optimizations

| Optimization | Time for 3,235 counties | Speedup |
|--------------|------------------------|---------|
| Current (no cache) | 12.3 hours | 1x |
| Current (with cache) | 1.5 hours | 8x |
| + Pre-computed baselines | 1.5 hours | 8x |
| + Optimized data loading | 0.5 hours | 25x |
| + GPU acceleration | 0.1 hours | 120x |
| + Cloud architecture | < 0.05 hours | 250x |

## Quick Wins for Tomorrow

1. Run baseline pre-computation overnight:
   ```bash
   python precompute_baselines.py > baseline_computation.log 2>&1 &
   ```

2. Increase batch size for better file handle sharing:
   ```python
   processor.process_parallel_optimized(
       counties_per_batch=100  # Instead of 50
   )
   ```

3. Use SSD for cache directory:
   ```python
   processor = OptimizedParallelProcessor(
       cache_dir="/fast/ssd/climate_cache"
   )
   ```

These optimizations will enable processing all US counties in under 2 hours on a single machine, or under 15 minutes with cloud resources.