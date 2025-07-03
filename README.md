# 🌡️ Climate Zarr Toolkit

A powerful, **interactive CLI toolkit** for processing climate data with guided wizards, smart prompts, and beautiful user experiences. Features cutting-edge NetCDF to Zarr conversion and county-level statistical analysis.

## 🚀 Main Features

- **🗜️ NetCDF → Zarr Conversion**: Convert multiple NetCDF files to optimized Zarr format with compression
- **📈 County Statistics**: Calculate detailed climate statistics by county/region with parallel processing
- **🗺️ Regional Clipping**: Built-in support for US regions (CONUS, Alaska, Hawaii, etc.)
- **🌡️ Multiple Variables**: Support for precipitation, temperature, and extreme weather analysis
- **⚡ Modern Performance**: Leverages Dask, parallel processing, and modern data formats
- **🎨 Beautiful CLI**: Rich-powered interface with progress bars and beautiful output

## ✨ Interactive Features

- **🧙‍♂️ Interactive Wizard**: Complete guided experience for beginners and experts
- **🎯 Smart Prompts**: Intelligent parameter suggestions with beautiful selection menus
- **✅ Safety Confirmations**: Prevent accidental data loss with confirmation dialogs
- **📂 Smart File Detection**: Automatically discovers and suggests data sources
- **🗺️ Visual Region Selection**: Choose regions with descriptions and coverage details
- **🔬 Variable Picker**: Climate variable selection with tooltips and explanations
- **⚡ Performance Tuning**: Interactive optimization suggestions for your workflow

## 🎮 Interactive vs Command-Line Modes

This toolkit offers **three ways** to work with climate data:

### 🧙‍♂️ Wizard Mode - **Best for Beginners**
Complete guided experience with step-by-step instructions:

```bash
# Launch the interactive wizard
python climate_cli.py wizard

# The wizard will guide you through:
# 1. ✨ Choose your workflow (convert, analyze, or both)
# 2. 📁 Smart file/directory selection
# 3. 🗺️ Regional clipping with visual descriptions
# 4. 🔬 Climate variable selection with explanations
# 5. ⚙️ Performance optimization suggestions
# 6. ✅ Safety confirmations before processing
# 7. 📊 Beautiful results summary
```

### 🎯 Interactive Mode - **Best for Daily Use**
Individual commands with intelligent prompting:

```bash
# Interactive NetCDF → Zarr conversion
python climate_cli.py create-zarr
# Prompts: Select files → Output name → Region? → Compression?

# Interactive county statistics
python climate_cli.py county-stats  
# Prompts: Zarr path → Region → Variable → Threshold → Output file
```

### ⚡ Command-Line Mode - **Best for Automation**
Traditional CLI for scripts and automation:

```bash
# Non-interactive mode (disable prompts)
python climate_cli.py create-zarr data/ -o output.zarr --region conus --interactive false
python climate_cli.py county-stats data.zarr conus -v pr -t 25.4 --interactive false
```

## 📦 Data Requirements

**Important**: This toolkit requires you to provide your own data files. The repository does not include large data files to keep it lightweight and fast to clone.

### Required Data Files

1. **🌡️ Climate Data**: NetCDF files with climate variables (precipitation, temperature, etc.)
   - Sources: [NASA NEX-GDDP](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6), [ECMWF ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels), [NOAA Climate Data](https://www.ncei.noaa.gov/data/)
   - Format: CF-compliant NetCDF with time, lat, lon dimensions
   - Place in: `data/` directory

2. **🗺️ US County Boundaries**: Census TIGER/Line county shapefiles
   - **Quick Setup**: See detailed instructions in [`utils/README.md`](utils/README.md)
   - **Source**: [US Census Bureau TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
   - **Processing**: Use our `utils/split_counties_by_region.py` script to split by regions

### Quick Data Setup

```bash
# 1. Place your NetCDF files in the data directory
mkdir -p data/
# Copy your .nc files to data/

# 2. Download and prepare county shapefiles (see utils/README.md for details)
cd utils/
# Follow instructions in utils/README.md to download and split shapefiles
python split_counties_by_region.py

# 3. You're ready to go!
cd ..
python climate_cli.py wizard
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mihiarc/climate-zarr
cd climate-zarr

# Install dependencies (using uv - the modern Python package manager)
uv install

# Or install in editable mode
uv pip install -e .
```

### Get Started in 30 Seconds

```bash
# 🧙‍♂️ NEW! Start with the interactive wizard (recommended)
python climate_cli.py wizard

# Or explore individual commands
python climate_cli.py --help          # See all commands
python climate_cli.py info            # Check your data
python climate_cli.py list-regions    # See available regions

# 🎮 Try interactive mode
python climate_cli.py create-zarr     # Interactive conversion
python climate_cli.py county-stats    # Interactive analysis
```

## 📖 Commands Reference

### 🧙‍♂️ **Wizard Mode** - Complete Guided Experience

```bash
python climate_cli.py wizard
# or
python climate_cli.py interactive
```

**Perfect for:**
- 🎓 Learning the toolkit
- 🔄 Complete end-to-end workflows  
- �� Complex multi-step analyses
- 🧠 Understanding best practices

### 🗜️ **Create Zarr from NetCDF**

**Interactive Mode (Recommended):**
```bash
python climate_cli.py create-zarr
# The CLI will guide you through:
# - 📂 File/directory selection
# - 📁 Output naming
# - 🗺️ Regional clipping options
# - 🗜️ Compression settings
```

**Command-Line Mode:**
```bash
# Basic conversion
python climate_cli.py create-zarr data/ -o precipitation.zarr

# Convert with region clipping (CONUS only)
python climate_cli.py create-zarr data/ -o conus_precip.zarr --region conus

# Custom chunking and compression
python climate_cli.py create-zarr data/ \
    -o optimized.zarr \
    --chunks "time=365,lat=180,lon=360" \
    --compression zstd \
    --compression-level 7

# Non-interactive mode for scripts
python climate_cli.py create-zarr data/ -o output.zarr --interactive false
```

**Options:**
- `--output, -o`: Output Zarr store path
- `--region, -r`: Clip to specific region (conus, alaska, hawaii, etc.)
- `--concat-dim, -d`: Dimension to concatenate along (default: time)
- `--chunks, -c`: Custom chunk sizes
- `--compression`: Algorithm (default, zstd, zlib, gzip)
- `--compression-level`: Level 1-9 (default: 5)
- `--interactive, -i`: Enable/disable interactive prompts (default: true)

### 📈 **Calculate County Statistics**

**Interactive Mode (Recommended):**
```bash
python climate_cli.py county-stats
# The CLI will guide you through:
# - 📁 Zarr dataset selection  
# - 🗺️ Region selection with descriptions
# - 🔬 Climate variable picker with tooltips
# - 🎯 Threshold configuration
# - ⚡ Performance settings
```

**Command-Line Mode:**
```bash
# Basic precipitation analysis for CONUS
python climate_cli.py county-stats precipitation.zarr conus -v pr -t 25.4

# Temperature analysis for Alaska with more workers
python climate_cli.py county-stats temperature.zarr alaska \
    -v tas \
    --workers 8 \
    -o alaska_temp_stats.csv

# Extreme heat analysis for Hawaii
python climate_cli.py county-stats extremes.zarr hawaii \
    -v tasmax \
    -t 90 \
    --scenario future

# Using distributed processing
python climate_cli.py county-stats large_dataset.zarr conus \
    -v pr \
    --distributed \
    --workers 16

# Non-interactive mode for scripts
python climate_cli.py county-stats data.zarr conus -v pr -t 25.4 --interactive false
```

**Options:**
- `--output, -o`: Output CSV file
- `--variable, -v`: Climate variable (pr, tas, tasmax, tasmin)
- `--scenario, -s`: Scenario name (default: historical)
- `--threshold, -t`: Threshold value for analysis
- `--workers, -w`: Number of worker processes
- `--distributed`: Use Dask distributed processing
- `--chunk-counties`: Process counties in chunks (default: True)
- `--interactive, -i`: Enable/disable interactive prompts (default: true)

### 🗺️ Available Regions

The toolkit supports these predefined regions:

| Region | Name | Coverage |
|--------|------|----------|
| `conus` | Continental US | 24.0°N to 50.0°N, -125.0°E to -66.0°E |
| `alaska` | Alaska | 54.0°N to 72.0°N, -180.0°E to -129.0°E |
| `hawaii` | Hawaii | 18.0°N to 23.0°N, -162.0°E to -154.0°E |
| `guam` | Guam/MP | 13.0°N to 21.0°N, 144.0°E to 146.0°E |
| `puerto_rico` | Puerto Rico/USVI | 17.5°N to 18.6°N, -67.5°E to -64.5°E |
| `global` | Global | Full global coverage |

## 🔬 Supported Climate Variables

| Variable | Description | Units | Statistics Generated |
|----------|-------------|-------|---------------------|
| `pr` | Precipitation | mm/day | Total annual, days >25.4mm, mean daily, max daily, dry days |
| `tas` | Air Temperature | °C | Mean annual, min/max, range, std dev, freezing days, hot days |
| `tasmax` | Daily Maximum Temperature | °C | Mean annual max, extremes, hot days above threshold |
| `tasmin` | Daily Minimum Temperature | °C | Mean annual min, cold days, frost-free period |

## 📁 Project Structure

```
climate-zarr/
├── climate_cli.py              # 🎯 Interactive CLI tool (NEW!)
├── stack_nc_to_zarr.py         # NetCDF → Zarr conversion
├── calculate_county_stats.py   # County statistics processor
├── climate_config.py           # Configuration management
├── demo_cli.py                 # Interactive demo script
├── utils/
│   ├── split_counties_by_region.py  # County shapefile splitter
│   └── README.md               # Data preparation instructions
├── data/                       # 📁 NetCDF input files (user-provided)
├── regional_counties/          # 🗺️ County shapefiles by region (user-generated)
└── pyproject.toml             # Project dependencies
```

**Note**: `data/` and `regional_counties/` directories are not included in the repository. Users must:
1. Add their own NetCDF climate data to `data/`
2. Follow [`utils/README.md`](utils/README.md) to download and prepare county shapefiles

## 🎯 Usage Examples

### 🧙‍♂️ **Complete Workflow with Wizard** (Recommended for beginners)

```bash
# Start the interactive wizard
python climate_cli.py wizard

# Follow the guided prompts:
# 1. "What would you like to do?" → Full pipeline
# 2. "Select your data source" → Choose data/ directory  
# 3. "Configure Zarr conversion" → CONUS region, ZSTD compression
# 4. "Configure county statistics" → Precipitation, 25.4mm threshold
# 5. Review settings and confirm
# 6. Watch beautiful progress bars and get comprehensive results!
```

### 🎯 **Interactive Command Workflow** (Daily use)

```bash
# 1. Check available data (always good to start here)
python climate_cli.py info

# 2. Interactive NetCDF → Zarr conversion
python climate_cli.py create-zarr
# Follow prompts: data/ → precipitation.zarr → CONUS → ZSTD

# 3. Interactive county statistics
python climate_cli.py county-stats  
# Follow prompts: precipitation.zarr → CONUS → pr → 25.4 → results.csv
```

### ⚡ **Command-Line Workflow** (Automation & scripts)

```bash
# 1. Check available data and regions
python climate_cli.py info
python climate_cli.py list-regions

# 2. Convert NetCDF to Zarr for CONUS region
python climate_cli.py create-zarr data/ \
    -o conus_precipitation.zarr \
    --region conus \
    --compression zstd \
    --interactive false

# 3. Calculate county precipitation statistics
python climate_cli.py county-stats conus_precipitation.zarr conus \
    -v pr \
    -t 25.4 \
    -o conus_precip_stats.csv \
    --workers 8 \
    --interactive false

# 4. Analyze temperature extremes for different regions
python climate_cli.py county-stats temperature.zarr alaska \
    -v tasmin \
    -o alaska_cold_stats.csv \
    --interactive false

python climate_cli.py county-stats temperature.zarr hawaii \
    -v tasmax \
    -t 32 \
    -o hawaii_heat_stats.csv \
    --interactive false
```

### 🎮 **Mixed Interactive & Command-Line**

```bash
# Use interactive mode for complex decisions, CLI for known parameters
python climate_cli.py create-zarr          # Interactive file/region selection
python climate_cli.py county-stats data.zarr conus -v pr  # Known dataset, interactive for other params
```

## 🛠️ Technical Details

### Modern Interactive Stack (2025)
- **CLI Framework**: Typer with Rich integration for beautiful output
- **Interactive Prompts**: Questionary for beautiful selection menus and confirmations
- **Data Processing**: xarray, dask, zarr (v3 ready)
- **Geospatial**: geopandas, rioxarray, pyogrio
- **Performance**: Parallel processing, chunked operations
- **Visualization**: Rich progress bars, tables, panels

### Interactive Features
- **Smart File Detection**: Automatically scans common directories (data/, input/, netcdf/)
- **Contextual Suggestions**: Intelligent defaults based on your data and previous choices
- **Error Recovery**: When something goes wrong, get interactive suggestions to fix it
- **Safety First**: Confirmation dialogs before potentially long-running or destructive operations
- **Progress Tracking**: Beautiful progress bars and real-time status updates

### Performance Features
- **Intelligent Chunking**: Automatically optimized for your data
- **Parallel Processing**: Multiprocessing + optional Dask distributed
- **Memory Efficient**: Chunked county processing for large datasets
- **Compression**: Multiple algorithms (zstd, zlib, gzip) with tunable levels

### Data Formats
- **Input**: NetCDF (.nc) files with CF conventions
- **Output**: Zarr v2/v3 stores, CSV statistics
- **Coordinates**: Automatic handling of different longitude conventions

## 🎓 Interactive Learning Mode

### For New Users:
1. **Start with**: `python climate_cli.py wizard`
2. **Learn basics**: Follow the guided tour and explanations
3. **Practice**: Try interactive commands with `python climate_cli.py create-zarr`
4. **Advanced**: Move to command-line mode for automation

### For Experienced Users:
1. **Quick setup**: `python climate_cli.py create-zarr data/ -o output.zarr --region conus`
2. **Interactive help**: Use prompts when you need parameter suggestions
3. **Automation**: Use `--interactive false` for scripts and CI/CD

### For Developers:
1. **Study the code**: Modern CLI patterns with Typer + Rich + Questionary
2. **Extend features**: Add new interactive prompts and wizard steps
3. **Learn patterns**: Type hints, async processing, configuration management

## 🎮 Demo & Testing

```bash
# 🎬 Run the comprehensive interactive demo
python demo_cli.py

# 🧪 Test individual features
python climate_cli.py wizard           # Full wizard experience
python climate_cli.py create-zarr      # Interactive conversion
python climate_cli.py county-stats     # Interactive analysis
python climate_cli.py info             # System overview
python climate_cli.py --help           # See all commands
```

## 🤝 Contributing

This is a modern, educational toolkit showcasing 2025 best practices in **interactive CLI development**. Key patterns demonstrated:

- **🎨 Modern CLI Design**: Typer + Rich + Questionary for beautiful UX
- **🎯 Interactive UX Patterns**: Beautiful prompts, confirmations, selections
- **⚡ Performance Optimization**: Chunking, compression, parallel processing
- **🎯 CLI Design Excellence**: User-friendly command interfaces with rich feedback
- **🔄 Data Engineering**: Efficient conversion and processing pipelines
- **🌟 Open Source Integration**: Leveraging the latest ecosystem tools

**🎬 Start your journey**: `python climate_cli.py wizard`

---

*Built with modern 2025 tools: Python 3.10+, Typer, Rich, Questionary, xarray, Zarr, Dask, and more!*
