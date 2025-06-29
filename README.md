# NEX-GDDP Climate Data Processor

A Python toolkit for processing NEX-GDDP-CMIP6 climate data at the county level using xclim climate indicators.

## Overview

This project provides tools to:
- Extract climate data for US counties from NEX-GDDP-CMIP6 NetCDF files
- Calculate standardized climate indicators using the xclim library
- Process data in parallel for efficient computation
- Use fixed historical baselines for comparable percentile calculations

## Key Features

- **County-level extraction**: Uses US Census county shapefiles to extract spatially averaged climate data
- **Climate indicators**: Implements key extreme indices including:
  - `tx90p`: Days exceeding 90th percentile of maximum temperature
  - `tx_days_above_90F`: Days with maximum temperature above 90°F
  - `tn10p`: Days below 10th percentile of minimum temperature
  - `tn_days_below_32F`: Days with minimum temperature below 32°F (frost days)
  - `tg_mean`: Annual mean temperature
  - `days_precip_over_25.4mm`: Heavy precipitation days (>1 inch)
  - `precip_accumulation`: Total annual precipitation
- **Fixed baseline approach**: Uses 1980-2010 as the standard climatological baseline for percentile calculations
- **Parallel processing**: Efficiently processes multiple counties using multiprocessing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/claude_climate.git
cd claude_climate

# Install dependencies (using uv package manager)
uv pip install xarray geopandas regionmask xclim pandas numpy
```

## Project Structure

```
claude_climate/
├── src/                           # Main processing scripts
│   ├── parallel_xclim_processor_fixed.py  # Fixed baseline parallel processor
│   ├── parallel_xclim_processor.py        # Standard parallel processor
│   └── xclim_indicators_processor.py      # Sequential processor
├── tests/                         # Test scripts
│   ├── test_fixed_baseline.py
│   ├── test_xclim_indicators.py
│   └── ...
├── data/
│   └── shapefiles/               # US county boundary files
├── results/                      # Output CSV files
├── archive/                      # Archived/deprecated scripts
└── README.md
```

## Usage

### Basic Example

```python
from src.parallel_xclim_processor_fixed import ParallelXclimProcessorFixed

# Initialize processor with fixed baseline
processor = ParallelXclimProcessorFixed(
    counties_shapefile_path="data/shapefiles/tl_2024_us_county.shp",
    base_data_path="/path/to/NEX-GDDP-data",
    baseline_period=(1980, 2010)  # 30-year climatological baseline
)

# Process climate indicators
df = processor.process_xclim_parallel(
    scenarios=['historical', 'ssp245'],
    variables=['tas', 'tasmax', 'tasmin', 'pr'],
    historical_period=(2005, 2014),
    future_period=(2040, 2049),
    n_chunks=16  # Number of parallel processes
)

# Save results
df.to_csv("climate_indicators.csv", index=False)
```

### Testing

Run the test scripts to verify functionality:

```bash
python tests/test_fixed_baseline.py
```

## Data Requirements

1. **NEX-GDDP-CMIP6 Data**: Download from [NASA EarthData](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6)
   - Required variables: tas, tasmax, tasmin, pr
   - File naming convention: `{variable}_day_{model}_{scenario}_*.nc`

2. **County Shapefiles**: US Census TIGER/Line shapefiles
   - Download from [Census.gov](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
   - Required fields: GEOID, NAME, STATEFP

## Key Implementation Details

### Fixed Baseline Approach
The `parallel_xclim_processor_fixed.py` implements a fixed 1980-2010 baseline for calculating percentile thresholds. This ensures that metrics are comparable across:
- Different time periods (historical vs future)
- Different locations
- Different scenarios

### Coordinate System Handling
The processors automatically handle the longitude coordinate conversion from -180/180° to 0/360° format used by NEX-GDDP data:
```python
if lon_min < 0:
    lon_min = lon_min % 360
```

### Percentile Calculation
Percentiles are calculated as day-of-year values from the baseline period, then applied to calculate exceedances in analysis periods. Results are converted from counts to percentages for easier interpretation.

## Output Format

The processors generate CSV files with the following columns:
- `GEOID`: County FIPS code
- `NAME`: County name
- `STATE`: State FIPS code
- `scenario`: Climate scenario (historical, ssp245, etc.)
- `year`: Year of data
- `tx90p_percent`: Percentage of days exceeding 90th percentile of maximum temperature
- `tx_days_above_90F`: Count of days above 90°F
- `tn10p_percent`: Percentage of days below 10th percentile of minimum temperature
- `tn_days_below_32F`: Count of frost days
- `tg_mean_C`: Annual mean temperature (°C)
- `days_precip_over_25.4mm`: Count of heavy precipitation days
- `precip_accumulation_mm`: Total annual precipitation (mm)

## License

This project is licensed under the MIT License.

## Acknowledgments

- NEX-GDDP-CMIP6 data provided by NASA Climate Simulation Center
- xclim library for standardized climate indicator calculations
- US Census Bureau for county boundary data