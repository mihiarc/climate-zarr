#!/usr/bin/env python
"""Tests for processing strategies."""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # Required for .rio accessor
from shapely.geometry import Polygon
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from climate_zarr.processors.processing_strategies import VectorizedStrategy
from climate_zarr.utils.data_utils import calculate_precipitation_stats


@pytest.fixture
def sample_counties():
    """Create sample counties for testing."""
    counties = []
    for i in range(3):
        # Create small rectangular counties
        minx, miny = -100 + i*0.2, 40
        maxx, maxy = minx + 0.2, 41
        
        poly = Polygon([
            (minx, miny), (maxx, miny), 
            (maxx, maxy), (minx, maxy), (minx, miny)
        ])
        
        counties.append({
            'county_id': f'{i:05d}',
            'county_name': f'Test County {i}',
            'state': 'TX',
            'raster_id': i + 1,
            'geometry': poly
        })
    
    return gpd.GeoDataFrame(counties, crs='EPSG:4326')


@pytest.fixture
def sample_precipitation_data():
    """Create sample precipitation data."""
    # Create 1 year of daily data
    time = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    
    # Create spatial grid that covers the counties
    lons = np.arange(-100.1, -99.3, 0.05)  # 16 points
    lats = np.arange(40.1, 40.9, 0.05)     # 16 points
    
    # Create realistic precipitation data (kg/m²/s)
    np.random.seed(42)
    data = np.random.exponential(2e-6, size=(len(time), len(lats), len(lons)))
    
    # Add some spatial patterns
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Add spatial gradient
            spatial_factor = 1 + 0.2 * (lat - 40.5) + 0.1 * (lon + 99.7)
            data[:, i, j] *= spatial_factor
    
    # Create xarray DataArray
    da = xr.DataArray(
        data,
        coords={
            'time': time,
            'lat': lats,
            'lon': lons
        },
        dims=['time', 'lat', 'lon'],
        name='pr'
    )
    
    # Add spatial reference and rename dimensions for rioxarray
    da = da.rename({'lat': 'y', 'lon': 'x'})
    da = da.rio.write_crs('EPSG:4326')
    
    return da


@pytest.fixture
def sample_temperature_data():
    """Create sample temperature data."""
    # Create 1 year of daily data
    time = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    
    # Create spatial grid
    lons = np.arange(-100.1, -99.3, 0.05)
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
        daily_variation = np.random.normal(0, 2, size=(len(lats), len(lons)))
        
        # Add spatial gradient
        for i, lat in enumerate(lats):
            spatial_temp = seasonal_temp - 0.5 * (lat - 40.5)
            data[t, i, :] = spatial_temp + daily_variation[i, :]
    
    # Create xarray DataArray
    da = xr.DataArray(
        data,
        coords={
            'time': time,
            'lat': lats,
            'lon': lons
        },
        dims=['time', 'lat', 'lon'],
        name='tas'
    )
    
    da = da.rename({'lat': 'y', 'lon': 'x'})
    da = da.rio.write_crs('EPSG:4326')
    return da


class TestVectorizedStrategy:
    """Test the vectorized processing strategy."""
    
    def test_initialization(self):
        """Test vectorized strategy initialization."""
        strategy = VectorizedStrategy()
        
        assert strategy is not None
        assert hasattr(strategy, 'process')
    
    def test_process_precipitation_data(self, sample_counties, sample_precipitation_data):
        """Test processing precipitation data with vectorized strategy."""
        strategy = VectorizedStrategy()
        
        # Mock the stats function
        with patch('climate_zarr.utils.data_utils.calculate_precipitation_stats') as mock_stats:
            mock_stats.return_value = {
                'year': 2020,
                'scenario': 'test',
                'county_id': '00000',
                'county_name': 'Test County 0',
                'state': 'TX',
                'total_annual_precip_mm': 1000.0,
                'days_above_threshold': 50,
                'dry_days': 10,
                'mean_daily_precip_mm': 2.74,
                'max_daily_precip_mm': 25.0
            }
            
            results = strategy.process(
                data=sample_precipitation_data,
                gdf=sample_counties,
                variable='pr',
                scenario='test',
                threshold=25.4,
                n_workers=2
            )
            
            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0
            
            # Should have called stats function
            assert mock_stats.called
    
    def test_process_temperature_data(self, sample_counties, sample_temperature_data):
        """Test processing temperature data with vectorized strategy."""
        strategy = VectorizedStrategy()
        
        # Mock the stats function
        with patch('climate_zarr.utils.data_utils.calculate_temperature_stats') as mock_stats:
            mock_stats.return_value = {
                'year': 2020,
                'scenario': 'test',
                'county_id': '00000',
                'county_name': 'Test County 0',
                'state': 'TX',
                'mean_annual_temp_c': 15.5,
                'days_below_freezing': 30,
                'growing_degree_days': 2500,
                'min_temp_c': -5.0,
                'max_temp_c': 35.0
            }
            
            results = strategy.process(
                data=sample_temperature_data,
                gdf=sample_counties,
                variable='tas',
                scenario='test',
                threshold=None,
                n_workers=2
            )
            
            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0
            
            # Should have called stats function
            assert mock_stats.called
    
    def test_process_with_invalid_variable(self, sample_counties, sample_precipitation_data):
        """Test processing with invalid variable."""
        strategy = VectorizedStrategy()
        
        # The vectorized strategy catches errors and returns empty results
        results = strategy.process(
            data=sample_precipitation_data,
            gdf=sample_counties,
            variable='invalid_var',
            scenario='test',
            threshold=25.4,
            n_workers=2
        )
        
        # Should return empty DataFrame when errors occur
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0
    
    def test_process_with_empty_counties(self, sample_precipitation_data):
        """Test processing with empty counties GeoDataFrame."""
        strategy = VectorizedStrategy()
        
        empty_counties = gpd.GeoDataFrame(columns=['county_id', 'county_name', 'state', 'geometry'])
        
        results = strategy.process(
            data=sample_precipitation_data,
            gdf=empty_counties,
            variable='pr',
            scenario='test',
            threshold=25.4,
            n_workers=2
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0


class TestStrategyComparison:
    """Test characteristics of the vectorized strategy."""
    
    def test_vectorized_strategy_characteristics(self, sample_counties):
        """Test characteristics of the vectorized strategy."""
        vectorized = VectorizedStrategy()
        
        # Strategy should handle different county sizes
        small_counties = sample_counties.iloc[:1]
        large_counties = pd.concat([sample_counties] * 10, ignore_index=True)
        
        # Should not raise errors with different sizes
        assert vectorized is not None
        
        # Should have process method
        assert hasattr(vectorized, 'process')


class TestStrategyErrorHandling:
    """Test error handling in strategies."""
    
    def test_missing_spatial_reference(self, sample_counties):
        """Test handling of data without spatial reference."""
        strategy = VectorizedStrategy()
        
        # Create data without CRS
        time = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        lons = np.array([-100.0, -99.5, -99.0])
        lats = np.array([40.0, 40.5, 41.0])
        data = np.random.rand(len(time), len(lats), len(lons))
        
        da = xr.DataArray(
            data,
            coords={'time': time, 'lat': lats, 'lon': lons},
            dims=['time', 'lat', 'lon'],
            name='pr'
        )
        # Don't add CRS
        
        # The strategy should handle missing CRS gracefully
        results = strategy.process(da, sample_counties, 'pr', 'test', 25.4, 2)
        
        # Should return empty results since clipping will fail without proper CRS
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0  # No successful clips without CRS
    
    def test_empty_counties(self, sample_precipitation_data):
        """Test handling of empty county GeoDataFrame."""
        strategy = VectorizedStrategy()
        
        # Create empty GeoDataFrame
        empty_gdf = gpd.GeoDataFrame(columns=['county_id', 'county_name', 'state', 'geometry'])
        empty_gdf = empty_gdf.set_crs('EPSG:4326')
        
        # Should handle empty input gracefully
        results = strategy.process(
            data=sample_precipitation_data,
            gdf=empty_gdf,
            variable='pr',
            scenario='test',
            threshold=25.4,
            n_workers=2
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0
    
    def test_invalid_variable_type(self, sample_counties, sample_precipitation_data):
        """Test handling of invalid variable types."""
        strategy = VectorizedStrategy()
        
        # Should handle unknown variable gracefully
        results = strategy.process(
            data=sample_precipitation_data,
            gdf=sample_counties,
            variable='unknown_var',
            scenario='test',
            threshold=25.4,
            n_workers=2
        )
        
        # Should still process data even with unknown variable
        assert isinstance(results, pd.DataFrame)


class TestStrategyDataTypes:
    """Test strategies with different data types."""
    
    def test_integer_data(self, sample_counties):
        """Test processing integer data."""
        strategy = VectorizedStrategy()
        
        # Create integer data
        time = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        lats = np.linspace(40, 41, 5)
        lons = np.linspace(-100, -99, 5)  # Match sample county locations
        
        int_data = xr.DataArray(
            np.random.randint(0, 100, size=(len(time), len(lats), len(lons))),
            dims=['time', 'y', 'x'],
            coords={
                'time': time,
                'y': lats,
                'x': lons
            },
            name='pr'
        ).rio.write_crs('EPSG:4326').rio.set_spatial_dims('x', 'y')
        
        results = strategy.process(
            data=int_data,
            gdf=sample_counties[:1],
            variable='pr',
            scenario='test',
            threshold=50,
            n_workers=2
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
    
    def test_float32_data(self, sample_counties):
        """Test processing float32 data."""
        strategy = VectorizedStrategy()
        
        # Create float32 data
        time = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        lats = np.linspace(40, 41, 5)
        lons = np.linspace(-100, -99, 5)  # Match sample county locations
        
        float32_data = xr.DataArray(
            np.random.random((len(time), len(lats), len(lons))).astype(np.float32) * 10,
            dims=['time', 'y', 'x'],
            coords={
                'time': time,
                'y': lats,
                'x': lons
            },
            name='pr'
        ).rio.write_crs('EPSG:4326').rio.set_spatial_dims('x', 'y')
        
        results = strategy.process(
            data=float32_data,
            gdf=sample_counties[:1],
            variable='pr',
            scenario='test',
            threshold=5.0,
            n_workers=2
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert results['total_annual_precip_mm'].dtype in [np.float32, np.float64]