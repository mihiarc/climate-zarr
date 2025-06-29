# Migration Guide: Unified Parallel Processor

## Overview

The `parallel_xclim_processor_unified.py` combines features from both the original and fixed baseline versions into a single, configurable processor.

## Key Features

### 1. Flexible Baseline Options

The unified processor supports two modes:

- **Fixed Baseline Mode** (recommended for climate change analysis):
  - Uses a consistent historical period (e.g., 1980-2010) for all percentile calculations
  - Ensures comparability across different time periods and scenarios
  - This is the standard approach in climate science

- **Period-Specific Mode** (for period-to-period comparisons):
  - Calculates percentiles from each analysis period
  - Useful when comparing climate characteristics between periods

### 2. Migration Examples

#### From `parallel_xclim_processor.py` (Original)

```python
# Old code
from parallel_xclim_processor import ParallelXclimProcessor
processor = ParallelXclimProcessor(
    counties_shapefile_path="path/to/shapefile.shp",
    base_data_path="/path/to/climate/data"
)

# New code (equivalent behavior)
from parallel_xclim_processor_unified import ParallelXclimProcessor
processor = ParallelXclimProcessor(
    counties_shapefile_path="path/to/shapefile.shp",
    base_data_path="/path/to/climate/data",
    use_fixed_baseline=False  # Period-specific percentiles
)
```

#### From `parallel_xclim_processor_fixed.py`

```python
# Old code
from parallel_xclim_processor_fixed import ParallelXclimProcessorFixed
processor = ParallelXclimProcessorFixed(
    counties_shapefile_path="path/to/shapefile.shp",
    base_data_path="/path/to/climate/data",
    baseline_period=(1980, 2010)
)

# New code (equivalent behavior)
from parallel_xclim_processor_unified import ParallelXclimProcessor
processor = ParallelXclimProcessor(
    counties_shapefile_path="path/to/shapefile.shp",
    base_data_path="/path/to/climate/data",
    baseline_period=(1980, 2010),
    use_fixed_baseline=True  # Fixed baseline percentiles
)
```

## API Differences

### Constructor Parameters

The unified processor adds two new parameters:

- `baseline_period`: Optional tuple (start_year, end_year) for baseline
  - Defaults to (1980, 2010) when `use_fixed_baseline=True`
  - Ignored when `use_fixed_baseline=False`

- `use_fixed_baseline`: Boolean flag to control percentile calculation method
  - `True`: Use fixed baseline period for all percentiles
  - `False`: Calculate percentiles from each analysis period

### Method Signatures

All other method signatures remain the same. The `process_xclim_parallel()` method works identically in all versions.

## Backward Compatibility

The unified processor maintains full backward compatibility:

1. **Default behavior matches the fixed baseline version** (recommended approach)
2. **All output formats remain identical**
3. **Method signatures are preserved**

## Recommendations

1. **For Climate Change Analysis**: Use fixed baseline mode
   ```python
   processor = ParallelXclimProcessor(
       counties_shapefile_path="...",
       base_data_path="...",
       baseline_period=(1980, 2010),
       use_fixed_baseline=True
   )
   ```

2. **For Period Comparisons**: Use period-specific mode
   ```python
   processor = ParallelXclimProcessor(
       counties_shapefile_path="...",
       base_data_path="...",
       use_fixed_baseline=False
   )
   ```

## Deprecation Notice

The individual processors (`parallel_xclim_processor.py` and `parallel_xclim_processor_fixed.py`) will be deprecated in favor of the unified processor. Please update your code to use `parallel_xclim_processor_unified.py`.