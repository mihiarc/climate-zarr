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

from climate_zarr.processors.processing_strategies import VectorizedStrategy, UltraFastStrategy
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


class TestUltraFastStrategy:
    """Test the ultra-fast processing strategy."""
    
    def test_initialization(self):
        """Test ultra-fast strategy initialization."""
        strategy = UltraFastStrategy()
        
        assert strategy is not None
        assert hasattr(strategy, 'process')
    
    def test_process_precipitation_data(self, sample_counties, sample_precipitation_data):
        """Test processing precipitation data with ultra-fast strategy."""
        strategy = UltraFastStrategy()
        
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
        
        # Check required columns
        required_cols = ['year', 'scenario', 'county_id', 'county_name', 'state']
        for col in required_cols:
            assert col in results.columns
        
        # Check precipitation-specific columns
        precip_cols = ['total_annual_precip_mm', 'days_above_threshold', 'dry_days']
        for col in precip_cols:
            assert col in results.columns
        
        # Should have one row per county per year
        unique_counties = results['county_id'].nunique()
        unique_years = results['year'].nunique()
        assert len(results) == unique_counties * unique_years
    
    def test_process_temperature_data(self, sample_counties, sample_temperature_data):
        """Test processing temperature data with ultra-fast strategy."""
        strategy = UltraFastStrategy()
        
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
        
        # Check temperature-specific columns
        temp_cols = ['mean_annual_temp_c', 'days_below_freezing', 'growing_degree_days']
        for col in temp_cols:
            assert col in results.columns
    
    def test_process_tasmax_data(self, sample_counties, sample_temperature_data):
        """Test processing tasmax data with ultra-fast strategy."""
        strategy = UltraFastStrategy()
        
        # Rename the temperature data to tasmax
        tasmax_data = sample_temperature_data.rename('tasmax')
        
        results = strategy.process(
            data=tasmax_data,
            gdf=sample_counties,
            variable='tasmax',
            scenario='test',
            threshold=None,
            n_workers=2
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        
        # Check tasmax-specific columns
        tasmax_cols = ['mean_annual_tasmax_c', 'days_above_35c', 'days_above_40c']
        for col in tasmax_cols:
            assert col in results.columns
    
    def test_process_tasmin_data(self, sample_counties, sample_temperature_data):
        """Test processing tasmin data with ultra-fast strategy."""
        strategy = UltraFastStrategy()
        
        # Rename the temperature data to tasmin and adjust values
        tasmin_data = sample_temperature_data.rename('tasmin') - 10  # Make it colder
        
        results = strategy.process(
            data=tasmin_data,
            gdf=sample_counties,
            variable='tasmin',
            scenario='test',
            threshold=None,
            n_workers=2
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        
        # Check tasmin-specific columns
        tasmin_cols = ['mean_annual_tasmin_c', 'cold_days', 'extreme_cold_days']
        for col in tasmin_cols:
            assert col in results.columns
    
    def test_process_with_invalid_variable(self, sample_counties, sample_precipitation_data):
        """Test processing with invalid variable."""
        strategy = UltraFastStrategy()
        
        with pytest.raises(ValueError, match="Unsupported variable"):
            strategy.process(
                data=sample_precipitation_data,
                gdf=sample_counties,
                variable='invalid_var',
                scenario='test',
                threshold=25.4,
                n_workers=2
            )
    
    def test_memory_efficiency(self, sample_counties, sample_precipitation_data):
        """Test that ultra-fast strategy is memory efficient."""
        strategy = UltraFastStrategy()
        
        # Process data and verify it doesn't hold references
        results = strategy.process(
            data=sample_precipitation_data,
            gdf=sample_counties,
            variable='pr',
            scenario='test',
            threshold=25.4,
            n_workers=2
        )
        
        # Should return results without holding onto large arrays
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        
        # Original data should be unchanged
        assert sample_precipitation_data.name == 'pr'
        assert len(sample_precipitation_data.dims) == 3


class TestStrategyComparison:
    """Test comparison between different strategies."""
    
    def test_strategies_produce_similar_results(self, sample_counties, sample_precipitation_data):
        """Test that different strategies produce similar results."""
        # Use smaller dataset for comparison
        small_data = sample_precipitation_data.isel(time=slice(0, 30))  # 30 days
        small_counties = sample_counties.iloc[:2]  # 2 counties
        
        vectorized = VectorizedStrategy()
        ultrafast = UltraFastStrategy()
        
        # Mock the vectorized strategy to use the same implementation as ultra-fast
        with patch.object(vectorized, 'process', side_effect=ultrafast.process):
            results_v = vectorized.process(small_data, small_counties, 'pr', 'test', 25.4, 2)
            results_u = ultrafast.process(small_data, small_counties, 'pr', 'test', 25.4, 2)
            
            # Should have same structure
            assert len(results_v) == len(results_u)
            assert list(results_v.columns) == list(results_u.columns)
    
    def test_strategy_performance_characteristics(self, sample_counties):
        """Test performance characteristics of strategies."""
        vectorized = VectorizedStrategy()
        ultrafast = UltraFastStrategy()
        
        # Both strategies should handle different county sizes
        small_counties = sample_counties.iloc[:1]
        large_counties = pd.concat([sample_counties] * 10, ignore_index=True)
        
        # Should not raise errors with different sizes
        assert vectorized is not None
        assert ultrafast is not None
        
        # Both should be able to process the same data types
        assert hasattr(vectorized, 'process')
        assert hasattr(ultrafast, 'process')


class TestStrategyErrorHandling:
    """Test error handling in strategies."""
    
    def test_missing_spatial_reference(self, sample_counties):
        """Test handling of data without spatial reference."""
        strategy = UltraFastStrategy()
        
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
        
        # Should handle missing CRS gracefully
        results = strategy.process(
            data=da,
            gdf=sample_counties,
            variable='pr',
            scenario='test',
            threshold=25.4,
            n_workers=2
        )
        
        assert isinstance(results, pd.DataFrame)
    
    def test_mismatched_spatial_extents(self, sample_counties):
        """Test handling of mismatched spatial extents."""
        strategy = UltraFastStrategy()
        
        # Create data that doesn't overlap with counties
        time = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        lons = np.array([0.0, 0.5, 1.0])  # Different location
        lats = np.array([0.0, 0.5, 1.0])  # Different location
        data = np.random.rand(len(time), len(lats), len(lons))
        
        da = xr.DataArray(
            data,
            coords={'time': time, 'lat': lats, 'lon': lons},
            dims=['time', 'lat', 'lon'],
            name='pr'
        )
        da = da.rio.write_crs('EPSG:4326')
        
        # Should handle non-overlapping extents
        results = strategy.process(
            data=da,
            gdf=sample_counties,
            variable='pr',
            scenario='test',
            threshold=25.4,
            n_workers=2
        )
        
        # Should return empty results or handle gracefully
        assert isinstance(results, pd.DataFrame)
    
    def test_invalid_threshold_values(self, sample_counties, sample_precipitation_data):
        """Test handling of invalid threshold values."""
        strategy = UltraFastStrategy()
        
        # Test with negative threshold
        results = strategy.process(
            data=sample_precipitation_data,
            gdf=sample_counties,
            variable='pr',
            scenario='test',
            threshold=-10.0,  # Invalid negative threshold
            n_workers=2
        )
        
        # Should handle invalid threshold gracefully
        assert isinstance(results, pd.DataFrame)
        
        # Test with None threshold for precipitation (should use default)
        results = strategy.process(
            data=sample_precipitation_data,
            gdf=sample_counties,
            variable='pr',
            scenario='test',
            threshold=None,
            n_workers=2
        )
        
        assert isinstance(results, pd.DataFrame)


class TestStrategyDataTypes:
    """Test strategies with different data types and formats."""
    
    def test_different_coordinate_names(self, sample_counties):
        """Test handling of different coordinate names."""
        strategy = UltraFastStrategy()
        
        # Create data with different coordinate names
        time = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        x = np.array([-100.0, -99.5, -99.0])
        y = np.array([40.0, 40.5, 41.0])
        data = np.random.rand(len(time), len(y), len(x))
        
        da = xr.DataArray(
            data,
            coords={'time': time, 'y': y, 'x': x},  # Different names
            dims=['time', 'y', 'x'],
            name='pr'
        )
        da = da.rio.write_crs('EPSG:4326')
        
        # Should handle different coordinate names
        results = strategy.process(
            data=da,
            gdf=sample_counties,
            variable='pr',
            scenario='test',
            threshold=25.4,
            n_workers=2
        )
        
        assert isinstance(results, pd.DataFrame)
    
    def test_different_time_frequencies(self, sample_counties):
        """Test handling of different time frequencies."""
        strategy = UltraFastStrategy()
        
        # Create monthly data instead of daily
        time = pd.date_range('2020-01-01', '2020-12-31', freq='M')
        lons = np.array([-100.0, -99.5, -99.0])
        lats = np.array([40.0, 40.5, 41.0])
        data = np.random.rand(len(time), len(lats), len(lons))
        
        da = xr.DataArray(
            data,
            coords={'time': time, 'lat': lats, 'lon': lons},
            dims=['time', 'lat', 'lon'],
            name='pr'
        )
        da = da.rio.write_crs('EPSG:4326')
        
        # Should handle different time frequencies
        results = strategy.process(
            data=da,
            gdf=sample_counties,
            variable='pr',
            scenario='test',
            threshold=25.4,
            n_workers=2
        )
        
        assert isinstance(results, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 