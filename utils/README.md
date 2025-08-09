# 🗺️ County Shapefile Preparation Utilities

This directory contains utilities for preparing US county boundary data for use with the Climate Zarr Toolkit. The tools help you download, process, and split US Census county shapefiles into regional files optimized for climate analysis.

## 📋 Quick Setup Guide

### Step 1: Download US County Shapefiles

**Option A: Direct Download (Recommended)**
```bash
# Download the latest US county boundaries from Census Bureau
wget https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip

# Extract the shapefile
unzip tl_2024_us_county.zip
```

**Option B: Manual Download**
1. Visit [US Census TIGER/Line Files](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
2. Select "2024" (or latest year) → "Counties (and equivalent)"
3. Download "tl_2024_us_county.zip"
4. Extract to get shapefile components

### Step 2: Split Counties by Climate Regions

```bash
# Run the modern county splitter
python split_counties_by_region.py

# This creates regional shapefiles in ../regional_counties/:
# - conus_counties.shp      (Continental US)
# - alaska_counties.shp     (Alaska)  
# - hawaii_counties.shp     (Hawaii)
# - puerto_rico_counties.shp (Puerto Rico & USVI)
# - guam_counties.shp       (Guam & N. Mariana)
# - other_counties.shp      (Other territories)
# - regional_summary.csv    (Statistics report)
```

### Step 3: Verify Results

```bash
# Check that regional files were created successfully
ls -la ../regional_counties/

# View the summary report
cat ../regional_counties/regional_summary.csv
```

## 📖 Detailed Instructions

### Understanding US County Data

The US Census Bureau provides county boundary data through the **TIGER/Line** program:

- **Format**: Shapefile (.shp, .dbf, .prj, .shx files)
- **Coverage**: All 3,234 US counties and county-equivalents
- **Includes**: All 50 states + DC + territories (Puerto Rico, Guam, etc.)
- **Update Frequency**: Annual updates
- **Coordinate System**: WGS84 (EPSG:4326)

### Available Years and Sources

| Year | Direct Download | Notes |
|------|----------------|-------|
| **2024** | [tl_2024_us_county.zip](https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip) | **Recommended** - Latest boundaries |
| 2023 | [tl_2023_us_county.zip](https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip) | Previous year |
| 2022 | [tl_2022_us_county.zip](https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip) | Stable version |

### Alternative Data Sources

**State-by-State Downloads**
```bash
# Download individual state counties (example for California)
wget https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_06_county.zip

# State FIPS codes: 01=AL, 02=AK, 04=AZ, 05=AR, 06=CA, 08=CO, etc.
```

**Cartographic Boundary Files** (Simplified, smaller files)
```bash
# Download simplified boundaries (good for visualization)
wget https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_20m.zip
```

## 🛠️ Using split_counties_by_region.py

### Script Overview

The `split_counties_by_region.py` script is a modern Python tool that:

✨ **Automatically splits** US counties into climate-relevant regions  
🌍 **Uses precise geographic boundaries** for each region  
⚡ **Leverages modern geospatial tools** (pyogrio, geopandas)  
📊 **Generates summary statistics** and reports  
🎨 **Beautiful terminal output** with Rich progress bars  

### Basic Usage

```bash
# Default usage (looks for tl_2024_us_county.shp)
python split_counties_by_region.py

# The script will:
# 1. 📁 Load the county shapefile
# 2. 🌍 Calculate county centroids 
# 3. 🗺️ Assign regions based on lat/lon boundaries
# 4. 💾 Save regional shapefiles to ../regional_counties/
# 5. 📊 Generate summary report
```

### Advanced Configuration

**Custom Input File**
```bash
# Modify the script to use a different shapefile
# Edit line 188 in split_counties_by_region.py:
splitter = ModernCountySplitter(
    shapefile_path="your_custom_county_file.shp",  # Change this
    output_dir="regional_counties"
)
```

**Custom Output Directory**
```bash
# Change output directory in the script:
splitter = ModernCountySplitter(
    shapefile_path="tl_2024_us_county.shp",
    output_dir="my_custom_regions"  # Change this
)
```

### Regional Boundaries

The script uses these precise geographic boundaries:

| Region | Latitude Range | Longitude Range | Counties |
|--------|---------------|-----------------|----------|
| **CONUS** | 24.0°N to 50.0°N | -125.0°W to -66.0°W | ~3,100 |
| **Alaska** | 54.0°N to 72.0°N | -180.0°W to -129.0°W | 30 |
| **Hawaii** | 18.0°N to 23.0°N | -162.0°W to -154.0°W | 5 |
| **Puerto Rico** | 17.5°N to 18.6°N | -67.5°W to -64.5°W | 78 |
| **Guam** | 13.0°N to 21.0°N | 144.0°E to 146.0°E | 5 |
| **Other** | Outside above ranges | - | Remaining |

## 📊 Output Files

### Regional Shapefiles

Each region gets a complete shapefile set:

```bash
regional_counties/
├── conus_counties.shp        # Continental US counties
├── conus_counties.dbf        # Attribute data
├── conus_counties.prj        # Projection info  
├── conus_counties.shx        # Spatial index
├── alaska_counties.shp       # Alaska counties
├── alaska_counties.dbf
├── alaska_counties.prj
├── alaska_counties.shx
├── hawaii_counties.shp       # Hawaii counties
├── [... similar for other regions ...]
└── regional_summary.csv      # Summary statistics
```

### Summary Report

The `regional_summary.csv` contains:

| Column | Description | Example |
|--------|-------------|---------|
| `region` | Region identifier | `conus` |
| `display_name` | Human-readable name | `Continental US` |
| `county_count` | Number of counties | `3067` |
| `lat_range` | Latitude coverage | `24.0 to 50.0` |
| `lon_range` | Longitude coverage | `-125.0 to -66.0` |
| `area_km2` | Total area in km² | `7834539.2` |

## 🔧 Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
# Install required packages
pip install geopandas rich

# Or with the climate-zarr environment
cd ..
uv install
```

**Shapefile Not Found**
```bash
# Make sure you've downloaded and extracted the county shapefile
ls -la tl_2024_us_county.*

# Should show:
# tl_2024_us_county.shp
# tl_2024_us_county.dbf  
# tl_2024_us_county.prj
# tl_2024_us_county.shx
```

**Permission Errors**
```bash
# Make sure the script is executable
chmod +x split_counties_by_region.py

# And the output directory is writable
mkdir -p ../regional_counties
```

**Memory Issues** (Large shapefiles)
```bash
# For very large files, you might need more memory
# Monitor with:
python -c "
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB')
"
```

### Validation

**Check Output Quality**
```bash
# Verify regional files were created
python -c "
import geopandas as gpd
import pandas as pd

# Load and check one region
conus = gpd.read_file('../regional_counties/conus_counties.shp')
print(f'CONUS counties: {len(conus)}')
print(f'Columns: {list(conus.columns)}')
print(f'CRS: {conus.crs}')

# Check summary
summary = pd.read_csv('../regional_counties/regional_summary.csv')
print('\nRegional Summary:')
print(summary[['region', 'county_count', 'area_km2']])
"
```

**Visual Verification** (Optional)
```bash
# Quick plot to verify regions look correct
python -c "
import geopandas as gpd
import matplotlib.pyplot as plt

# Load all regions and plot
fig, ax = plt.subplots(figsize=(15, 10))

colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
regions = ['conus', 'alaska', 'hawaii', 'puerto_rico', 'guam', 'other']

for i, region in enumerate(regions):
    try:
        gdf = gpd.read_file(f'../regional_counties/{region}_counties.shp')
        gdf.plot(ax=ax, color=colors[i], alpha=0.7, label=region)
    except:
        print(f'Skipping {region} (file not found)')

ax.legend()
plt.title('US Counties by Climate Region')
plt.savefig('county_regions_map.png', dpi=150, bbox_inches='tight')
print('Saved verification map to county_regions_map.png')
"
```

## 🎯 Integration with Climate Zarr Toolkit

Once you've prepared the regional shapefiles, they're ready for use with the main climate analysis tools:

```bash
# Return to main directory
cd ..

# Use your prepared regional counties
python climate_cli.py county-stats my_data.zarr conus -v pr -t 25.4

# The toolkit will automatically find your regional_counties/ files!
```

## 🔄 Updating County Data

**Annual Updates**
```bash
# Download latest county boundaries (run annually)
rm -f tl_*_us_county.*
wget https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip
unzip tl_2024_us_county.zip

# Re-split with latest boundaries
python split_counties_by_region.py
```

**Backup Previous Versions**
```bash
# Before updating, backup current regional files
cp -r ../regional_counties ../regional_counties_backup_$(date +%Y%m%d)

# Then proceed with update
```

## 📚 Additional Resources

- **📖 US Census TIGER/Line Documentation**: [census.gov/geo/maps-data/data/tiger-line.html](https://www.census.gov/geo/maps-data/data/tiger-line.html)
- **🌍 GeoPandas Documentation**: [geopandas.org](https://geopandas.org)
- **🗺️ Shapefile Format Specification**: [ESRI Shapefile Technical Description](https://www.esri.com/content/dam/esrisites/sitecore-archive/Files/Pdfs/library/whitepapers/pdfs/shapefile.pdf)
- **🎨 Rich Terminal Documentation**: [rich.readthedocs.io](https://rich.readthedocs.io)

---

*Part of the Climate Zarr Toolkit - Interactive 2025 Edition* 