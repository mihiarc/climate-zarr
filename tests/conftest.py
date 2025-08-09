#!/usr/bin/env python
"""Pytest configuration and shared fixtures for climate-zarr tests."""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # Required for .rio accessor
from shapely.geometry import Polygon
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test session."""
    temp_dir = tempfile.mkdtemp(prefix="climate_zarr_tests_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_counties_gdf():
    """Create a sample GeoDataFrame with test counties."""
    counties = []
    for i in range(5):
        # Create rectangular counties
        minx, miny = -100 + i*0.2, 40
        maxx, maxy = minx + 0.2, 41
        
        poly = Polygon([
            (minx, miny), (maxx, miny), 
            (maxx, maxy), (minx, maxy), (minx, miny)
        ])
        
        counties.append({
            'GEOID': f'{i:05d}',
            'NAME': f'Test County {i}',
            'STUSPS': 'TX',
            'geometry': poly
        })
    
    return gpd.GeoDataFrame(counties, crs='EPSG:4326')


@pytest.fixture
def sample_shapefile(sample_counties_gdf, temp_test_dir):
    """Create a sample shapefile for testing."""
    shapefile_path = Path(temp_test_dir) / "test_counties.shp"
    sample_counties_gdf.to_file(shapefile_path)
    return shapefile_path


@pytest.fixture
def sample_precipitation_xarray():
    """Create sample precipitation data as xarray DataArray."""
    # Create 1 year of daily data
    time = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    
    # Create spatial grid
    lons = np.arange(-100.1, -99.5, 0.05)  # 12 points
    lats = np.arange(40.1, 40.9, 0.05)     # 16 points
    
    # Create realistic precipitation data (kg/m²/s)
    np.random.seed(42)
    data = np.random.exponential(2e-6, size=(len(time), len(lats), len(lons)))
    
    # Add seasonal patterns
    for t in range(len(time)):
        day_of_year = time[t].dayofyear
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        data[t, :, :] *= seasonal_factor
    
    # Create xarray DataArray
    da = xr.DataArray(
        data,
        coords={'time': time, 'lat': lats, 'lon': lons},
        dims=['time', 'lat', 'lon'],
        name='pr'
    )
    
    # Add spatial reference and attributes
    da = da.rio.write_crs('EPSG:4326')
    da.attrs['units'] = 'kg/m2/s'
    da.attrs['long_name'] = 'precipitation'
    
    return da


@pytest.fixture
def sample_temperature_xarray():
    """Create sample temperature data as xarray DataArray."""
    # Create 1 year of daily data
    time = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    
    # Create spatial grid
    lons = np.arange(-100.1, -99.5, 0.05)
    lats = np.arange(40.1, 40.9, 0.05)
    
    # Create realistic temperature data (Kelvin)
    np.random.seed(123)
    base_temp = 283.15  # ~10°C
    
    data = np.zeros((len(time), len(lats), len(lons)))
    
    for t in range(len(time)):
        # Seasonal temperature variation
        day_of_year = time[t].dayofyear
        seasonal_temp = base_temp + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Add daily random variation
        daily_variation = np.random.normal(0, 3, size=(len(lats), len(lons)))
        
        # Add spatial gradient
        for i, lat in enumerate(lats):
            spatial_temp = seasonal_temp - 0.5 * (lat - 40.5)
            data[t, i, :] = spatial_temp + daily_variation[i, :]
    
    # Create xarray DataArray
    da = xr.DataArray(
        data,
        coords={'time': time, 'lat': lats, 'lon': lons},
        dims=['time', 'lat', 'lon'],
        name='tas'
    )
    
    # Add spatial reference and attributes
    da = da.rio.write_crs('EPSG:4326')
    da.attrs['units'] = 'K'
    da.attrs['long_name'] = 'air_temperature'
    
    return da


@pytest.fixture
def sample_precipitation_zarr(sample_precipitation_xarray, temp_test_dir):
    """Create sample precipitation zarr dataset."""
    zarr_path = Path(temp_test_dir) / "test_precipitation.zarr"
    ds = sample_precipitation_xarray.to_dataset()
    ds.to_zarr(zarr_path)
    return zarr_path


@pytest.fixture
def sample_temperature_zarr(sample_temperature_xarray, temp_test_dir):
    """Create sample temperature zarr dataset."""
    zarr_path = Path(temp_test_dir) / "test_temperature.zarr"
    ds = sample_temperature_xarray.to_dataset()
    ds.to_zarr(zarr_path)
    return zarr_path


@pytest.fixture
def sample_county_info():
    """Create sample county information dictionary."""
    return {
        'county_id': '12345',
        'county_name': 'Test County',
        'state': 'TX'
    }


@pytest.fixture
def sample_daily_precipitation():
    """Create sample daily precipitation values."""
    np.random.seed(42)
    return np.random.exponential(2.0, size=365)  # mm/day


@pytest.fixture
def sample_daily_temperature():
    """Create sample daily temperature values."""
    np.random.seed(123)
    # Create seasonal temperature pattern
    days = np.arange(365)
    base_temp = 15  # °C
    seasonal_temp = base_temp + 10 * np.sin(2 * np.pi * days / 365)
    daily_variation = np.random.normal(0, 3, size=365)
    return seasonal_temp + daily_variation


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to end-to-end tests
        if "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add unit marker to unit tests
        if any(name in item.nodeid for name in ["test_processors", "test_strategies", "test_utils"]):
            item.add_marker(pytest.mark.unit)


# Skip slow tests by default
def pytest_addoption(parser):
    """Add command line options for test selection."""
    parser.addoption(
        "--runslow", action="store_true", default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--runintegration", action="store_true", default=False,
        help="run integration tests"
    )


def pytest_runtest_setup(item):
    """Setup function to skip tests based on markers."""
    if "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
    
    if "integration" in item.keywords and not item.config.getoption("--runintegration"):
        pytest.skip("need --runintegration option to run")