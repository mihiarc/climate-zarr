"""
Pytest configuration and fixtures for Climate Zarr integration tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from datetime import datetime, timedelta


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="climate_zarr_test_")
    yield Path(temp_dir)
    # Cleanup after all tests
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_netcdf_files(test_data_dir):
    """Create sample NetCDF files with climate data."""
    nc_dir = test_data_dir / "netcdf"
    nc_dir.mkdir(exist_ok=True)
    
    # Create 3 years of monthly data
    files = []
    base_date = datetime(2020, 1, 1)
    
    for year in range(2020, 2023):
        # Create time coordinates for one year
        times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        
        # Create spatial coordinates (small grid for testing)
        lats = np.linspace(25, 45, 20)  # 20 latitude points
        lons = np.linspace(-120, -70, 25)  # 25 longitude points
        
        # Create sample data
        # Temperature (tas) - varies by latitude and time
        tas_data = np.random.normal(
            loc=20 - (lats[:, np.newaxis, np.newaxis] - 35) * 0.5,  # Temperature decreases with latitude
            scale=5,
            size=(len(lats), len(lons), len(times))
        )
        
        # Add seasonal variation
        day_of_year = times.dayofyear.values
        seasonal_factor = 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        tas_data += seasonal_factor[np.newaxis, np.newaxis, :]
        
        # Precipitation (pr) - random with some spatial correlation
        pr_data = np.maximum(
            np.random.gamma(2, 2, size=(len(lats), len(lons), len(times))),
            0
        )
        
        # Create dataset
        ds = xr.Dataset(
            {
                "tas": (["lat", "lon", "time"], tas_data, {"units": "degC", "long_name": "Temperature"}),
                "pr": (["lat", "lon", "time"], pr_data, {"units": "mm/day", "long_name": "Precipitation"}),
            },
            coords={
                "lat": ("lat", lats, {"units": "degrees_north", "long_name": "Latitude"}),
                "lon": ("lon", lons, {"units": "degrees_east", "long_name": "Longitude"}),
                "time": ("time", times),
            },
            attrs={
                "title": f"Test Climate Data {year}",
                "institution": "Climate Zarr Test Suite",
                "source": "Synthetic test data",
                "history": f"Created for testing on {datetime.now()}",
                "Conventions": "CF-1.8",
            }
        )
        
        # Save to NetCDF
        filename = nc_dir / f"climate_data_{year}.nc"
        ds.to_netcdf(filename, encoding={
            "tas": {"dtype": "float32", "zlib": True, "complevel": 4},
            "pr": {"dtype": "float32", "zlib": True, "complevel": 4},
        })
        files.append(filename)
    
    return files


@pytest.fixture(scope="session")
def sample_shapefile(test_data_dir):
    """Create a sample shapefile with mock county boundaries."""
    shp_dir = test_data_dir / "shapefiles"
    shp_dir.mkdir(exist_ok=True)
    
    # Create mock counties as rectangles
    counties = []
    county_data = [
        {"NAME": "Test County 1", "STATEFP": "06", "COUNTYFP": "001", "GEOID": "06001"},
        {"NAME": "Test County 2", "STATEFP": "06", "COUNTYFP": "002", "GEOID": "06002"},
        {"NAME": "Test County 3", "STATEFP": "06", "COUNTYFP": "003", "GEOID": "06003"},
    ]
    
    # Create geometries (non-overlapping boxes)
    for i, data in enumerate(county_data):
        # Create a 2x2 degree box
        minx = -120 + i * 3
        miny = 35 + i * 2
        geometry = box(minx, miny, minx + 2, miny + 2)
        counties.append({**data, "geometry": geometry})
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(counties, crs="EPSG:4326")
    
    # Save to shapefile
    shapefile_path = shp_dir / "test_counties.shp"
    gdf.to_file(shapefile_path)
    
    return shapefile_path


@pytest.fixture
def zarr_output_dir(test_data_dir):
    """Create a directory for Zarr output."""
    zarr_dir = test_data_dir / "zarr_output"
    zarr_dir.mkdir(exist_ok=True)
    yield zarr_dir
    # Cleanup is handled by test_data_dir fixture


@pytest.fixture
def stats_output_dir(test_data_dir):
    """Create a directory for statistics output."""
    stats_dir = test_data_dir / "stats_output"
    stats_dir.mkdir(exist_ok=True)
    yield stats_dir
    # Cleanup is handled by test_data_dir fixture


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing Typer apps."""
    from typer.testing import CliRunner
    return CliRunner()


@pytest.fixture
def mock_climate_config(test_data_dir):
    """Create a mock climate configuration."""
    from climate_zarr.climate_config import ClimateConfig, CompressionConfig, ChunkingConfig
    
    config = ClimateConfig(
        data_dir=str(test_data_dir / "netcdf"),
        output_dir=str(test_data_dir / "output"),
        compression=CompressionConfig(algorithm="zstd", level=3),
        chunking=ChunkingConfig(time=100, lat=10, lon=10),
    )
    return config


@pytest.fixture(scope="session")
def real_data_available():
    """Check if real data files are available."""
    data_dir = Path("data")
    regional_dir = Path("regional_counties")
    
    has_nc_files = data_dir.exists() and len(list(data_dir.glob("*.nc"))) > 0
    has_shapefiles = regional_dir.exists() and len(list(regional_dir.glob("*.shp"))) > 0
    
    return has_nc_files and has_shapefiles


@pytest.fixture(scope="session")
def sample_real_nc_file():
    """Get one real NetCDF file if available."""
    data_dir = Path("data")
    if not data_dir.exists():
        return None
    
    nc_files = sorted(data_dir.glob("pr_day_NorESM2-LM_historical_*.nc"))
    if nc_files:
        return nc_files[0]
    return None


@pytest.fixture(scope="session")
def real_shapefile_path():
    """Get path to a real shapefile if available."""
    # Try CONUS first as it's most commonly used
    conus_shp = Path("regional_counties/conus_counties.shp")
    if conus_shp.exists():
        return conus_shp
    
    # Try any shapefile
    regional_dir = Path("regional_counties")
    if regional_dir.exists():
        shapefiles = list(regional_dir.glob("*.shp"))
        if shapefiles:
            return shapefiles[0]
    
    return None