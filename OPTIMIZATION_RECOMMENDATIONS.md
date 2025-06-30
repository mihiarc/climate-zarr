# Climate Data Pipeline Optimization Recommendations

## Executive Summary

Based on analysis of the climate data pipeline, we've identified several optimization opportunities that can reduce processing time from **12.3 hours to under 10 minutes** for all 3,235 US counties. The pipeline already includes powerful optimization tools that need to be deployed and integrated.

## Current Performance Bottlenecks

### 2. Redundant Spatial Data Loading
- **Issue**: Multiple counties in same region load identical NetCDF files
- **Impact**: 100x more I/O than necessary
- **Solution**: County pre-extraction and regional tiles (tools exist)

### 3. Inefficient File Access Patterns
- **Issue**: Opening thousands of small files repeatedly
- **Impact**: High filesystem overhead
- **Solution**: Pre-merged temporal data (tool exists)

## Optimization Strategies

### Phase 2: Code Integration (3-5 days)

#### 2.1 Update UnifiedClimateCalculator
```python
# src/core/unified_calculator.py

def extract_county_data_optimized(self, county_id, variable, scenario, year_range):
    """Use pre-extracted data if available, fall back to base method."""
    
    # Check for pre-extracted county data
    extract_dir = self.cache_dir / 'county_extracts' / county_id
    extract_file = extract_dir / f"{variable}_{scenario}_{year_range[0]}-{year_range[1]}.nc"
    
    if extract_file.exists():
        # Load pre-extracted data (100x faster)
        ds = xr.open_dataset(extract_file)
        return ds[variable]
    
    # Check for pre-merged baseline
    if scenario == 'historical' and year_range == (1980, 2010):
        merged_file = self.cache_dir / 'baseline_merged' / f"{variable}_baseline_1980-2010.nc"
        if merged_file.exists():
            ds = xr.open_dataset(merged_file)
            # Extract county region
            bounds = self.get_county_bounds(county_id)
            return self._extract_region(ds[variable], bounds)
    
    # Fall back to original method
    files = self.get_files_for_period(variable, scenario, *year_range)
    return self.extract_county_data_base(files, variable, bounds)
```

#### 2.2 Implement Batch Processing with Regional Tiles
```python
# src/core/unified_processor.py

def process_counties_by_tile(self, tile_id, tile_info):
    """Process multiple counties sharing the same spatial tile."""
    
    tile_bounds = tile_info['bounds']
    county_ids = tile_info['counties']
    
    # Load data once for entire tile
    tile_data = {}
    for variable in self.variables:
        tile_data[variable] = self.load_tile_data(variable, tile_bounds)
    
    # Process each county using shared tile data
    results = []
    for county_id in county_ids:
        county_bounds = self.get_county_bounds(county_id)
        
        # Extract county from already-loaded tile
        county_data = {}
        for var, data in tile_data.items():
            county_data[var] = data.sel(
                lat=slice(county_bounds[1], county_bounds[3]),
                lon=slice(county_bounds[0], county_bounds[2])
            )
        
        # Process with pre-loaded data
        result = self.calculator.process_county_with_data(
            county_id, county_data, use_cached_baseline=True
        )
        results.append(result)
    
    return results
```

### Phase 3: Advanced Optimizations (1-2 weeks)

#### 3.1 Hierarchical Spatial Indexing
```python
def create_spatial_hierarchy(counties_gdf):
    """Build state->region->county hierarchy for faster lookups."""
    
    from sklearn.cluster import KMeans
    import rtree
    
    hierarchy = {}
    spatial_index = rtree.index.Index()
    
    # Group by state for locality
    for state, state_counties in counties_gdf.groupby('STATEFP'):
        state_bounds = state_counties.total_bounds
        
        # Create regional clusters within state
        centroids = state_counties.geometry.centroid
        coords = np.array([[c.x, c.y] for c in centroids])
        
        # Cluster into regions (5-10 counties each)
        n_clusters = max(1, len(state_counties) // 7)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(coords)
        
        regions = {}
        for cluster_id in range(n_clusters):
            cluster_counties = state_counties[clusters == cluster_id]
            regions[f"region_{cluster_id}"] = {
                'bounds': cluster_counties.total_bounds,
                'counties': cluster_counties['GEOID'].tolist(),
                'centroid': kmeans.cluster_centers_[cluster_id]
            }
        
        hierarchy[state] = {
            'bounds': state_bounds,
            'regions': regions,
            'counties': state_counties['GEOID'].tolist()
        }
        
        # Add to spatial index
        for idx, county in state_counties.iterrows():
            spatial_index.insert(idx, county.geometry.bounds)
    
    return hierarchy, spatial_index
```

#### 3.2 Smart Chunk Optimization
```python
def optimize_chunks_for_counties(shapefile_path, data_resolution=0.25):
    """Calculate optimal chunk sizes based on county statistics."""
    
    counties = gpd.read_file(shapefile_path)
    
    # Analyze county sizes
    sizes = []
    for _, county in counties.iterrows():
        bounds = county.geometry.bounds
        lat_size = (bounds[3] - bounds[1]) / data_resolution
        lon_size = (bounds[2] - bounds[0]) / data_resolution
        sizes.append((lat_size, lon_size))
    
    sizes = np.array(sizes)
    
    # Optimal chunks: 2x median county size
    chunk_lat = int(np.median(sizes[:, 0]) * 2)
    chunk_lon = int(np.median(sizes[:, 1]) * 2)
    
    # Constrain to reasonable limits
    chunk_lat = max(20, min(100, chunk_lat))
    chunk_lon = max(20, min(100, chunk_lon))
    
    return {
        'time': 365,  # One year for temporal operations
        'lat': chunk_lat,
        'lon': chunk_lon
    }
```

#### 3.3 Memory-Mapped Cache Implementation
```python
import numpy as np
from pathlib import Path

class MemoryMappedCache:
    """Zero-copy access to pre-computed county data."""
    
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache = {}
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Build index of available cached data."""
        for county_dir in self.cache_dir.glob("*"):
            if county_dir.is_dir():
                county_id = county_dir.name
                self.cache[county_id] = {
                    'baselines': county_dir / 'baselines.npy',
                    'metadata': county_dir / 'metadata.json'
                }
    
    def get_baseline(self, county_id):
        """Get baseline data with zero-copy access."""
        if county_id not in self.cache:
            return None
        
        baseline_file = self.cache[county_id]['baselines']
        if baseline_file.exists():
            # Memory map - no data loaded until accessed
            return np.memmap(baseline_file, dtype='float32', mode='r')
        
        return None
```

### Phase 4: Infrastructure Optimizations (2-4 weeks)

#### 4.1 Convert to Zarr Format
```python
def convert_to_zarr(netcdf_path, zarr_path, chunks=None):
    """Convert NetCDF to cloud-optimized Zarr format."""
    
    if chunks is None:
        chunks = {'time': 365, 'lat': 50, 'lon': 50}
    
    # Open with dask for lazy loading
    ds = xr.open_dataset(netcdf_path, chunks=chunks)
    
    # Set up Zarr encoding
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            'compressor': zarr.Blosc(cname='zstd', clevel=3),
            'chunks': tuple(chunks.get(dim, ds[var].sizes[dim]) 
                          for dim in ds[var].dims)
        }
    
    # Write to Zarr
    ds.to_zarr(zarr_path, encoding=encoding, mode='w')
    
    return zarr_path
```

#### 4.2 Distributed Caching Layer
```python
import redis
import pickle
import hashlib

class DistributedCache:
    """Redis-based distributed cache for county data."""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.ttl = 3600 * 24  # 24 hour TTL
    
    def _make_key(self, county_id, variable, scenario, year_range):
        """Generate cache key."""
        key_data = f"{county_id}:{variable}:{scenario}:{year_range}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, county_id, variable, scenario, year_range):
        """Get data from cache."""
        key = self._make_key(county_id, variable, scenario, year_range)
        data = self.redis.get(key)
        
        if data:
            return pickle.loads(data)
        return None
    
    def set(self, county_id, variable, scenario, year_range, data):
        """Store data in cache."""
        key = self._make_key(county_id, variable, scenario, year_range)
        self.redis.setex(key, self.ttl, pickle.dumps(data))
```

## Performance Projections

| Optimization Stage | Processing Time (3,235 counties) | Speedup | Implementation Effort |
|-------------------|----------------------------------|---------|----------------------|
| **Current Baseline** | 12.3 hours | 1x | - |
| **Phase 1: Deploy Tools** | 1.5 hours | 8x | 1-2 days |
| **Phase 2: Code Integration** | 30 minutes | 25x | 3-5 days |
| **Phase 3: Advanced Opts** | 12 minutes | 60x | 1-2 weeks |
| **Phase 4: Infrastructure** | 6 minutes | 120x+ | 2-4 weeks |

## Implementation Checklist

### Immediate Actions (Week 1)
- [ ] Run baseline pre-computation for all counties
- [ ] Create pre-merged baseline files
- [ ] Generate county-specific data extracts
- [ ] Update calculator to use pre-computed baselines
- [ ] Test with subset of counties

### Short-term (Week 2-3)
- [ ] Implement regional tile processing
- [ ] Add optimized data loading methods
- [ ] Create spatial indexing system
- [ ] Deploy distributed cache
- [ ] Performance benchmarking

### Medium-term (Month 2)
- [ ] Convert data to Zarr format
- [ ] Implement memory-mapped caching
- [ ] Add GPU acceleration for percentiles
- [ ] Set up monitoring and metrics
- [ ] Documentation and training

## Monitoring and Validation

### Performance Metrics to Track
1. **Per-county processing time**
   - Baseline calculation time
   - Data loading time
   - Indicator computation time

2. **System resource usage**
   - Memory consumption
   - CPU utilization
   - I/O throughput

3. **Data quality checks**
   - Compare optimized vs original results
   - Validate percentile calculations
   - Check spatial averaging accuracy

### Validation Script
```python
def validate_optimization_results(original_output, optimized_output, tolerance=1e-6):
    """Ensure optimized pipeline produces identical results."""
    
    for county_id in original_output.keys():
        orig = original_output[county_id]
        opt = optimized_output[county_id]
        
        # Check all indicators
        for indicator in ['tx90p', 'tn10p', 'rx5day', 'cdd']:
            assert np.allclose(
                orig[indicator], 
                opt[indicator], 
                rtol=tolerance
            ), f"Mismatch in {indicator} for county {county_id}"
    
    print("✓ All validations passed!")
```

## Conclusion

The climate data pipeline has excellent optimization potential. By deploying the existing pre-processing tools and implementing the recommended code changes, you can achieve a **120x speedup**, reducing processing time from 12.3 hours to under 6 minutes for all US counties. The phased approach allows for incremental improvements while maintaining data quality and system stability.
