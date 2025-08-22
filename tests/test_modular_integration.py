#!/usr/bin/env python
"""Integration tests for the modular climate-zarr architecture."""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path
from shapely.geometry import Polygon
import tempfile
import os

from climate_zarr import ModernCountyProcessor
from climate_zarr.processors import (
    PrecipitationProcessor,
    TemperatureProcessor,
    TasMaxProcessor,
    TasMinProcessor
)
from climate_zarr.processors.processing_strategies import VectorizedStrategy
from climate_zarr.utils import convert_units, get_time_information
from climate_zarr.utils.data_utils import (
    calculate_precipitation_stats,
    calculate_temperature_stats,
    calculate_tasmax_stats,
    calculate_tasmin_stats
)


@pytest.fixture
def sample_counties():
    """Create a sample GeoDataFrame with test counties."""
    # Create simple rectangular counties
    counties = []
    for i in range(3):
        for j in range(2):
            # Create 0.5 x 0.5 degree squares
            minx, miny = -100 + i*0.5, 40 + j*0.5
            maxx, maxy = minx + 0.5, miny + 0.5
            
            poly = Polygon([
                (minx, miny), (maxx, miny), 
                (maxx, maxy), (minx, maxy), (minx, miny)
            ])
            
            counties.append({
                'GEOID': f'{i:02d}{j:03d}',
                'NAME': f'County_{i}_{j}',
                'STUSPS': 'TX',
                'geometry': poly
            })
    
    gdf = gpd.GeoDataFrame(counties, crs='EPSG:4326')
    return gdf


@pytest.fixture
def sample_precipitation_data():
    """Create sample precipitation data as xarray Dataset."""
    # Create 2 years of daily data
    time = pd.date_range('2020-01-01', '2021-12-31', freq='D')
    
    # Create spatial grid covering the counties
    lons = np.arange(-100.25, -98.75, 0.1)  # 15 points
    lats = np.arange(40.25, 41.25, 0.1)    # 10 points
    
    # Create realistic precipitation data (kg/m²/s)
    np.random.seed(42)
    data = np.random.exponential(1e-6, size=(len(time), len(lats), len(lons)))
    
    # Add some spatial and temporal patterns
    for t in range(len(time)):
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Add seasonal pattern
                day_of_year = time[t].dayofyear
                seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365)
                
                # Add spatial gradient
                spatial_factor = 1 + 0.3 * (lat - 40.5) + 0.2 * (lon + 99.5)
                
                data[t, i, j] *= seasonal_factor * spatial_factor
    
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
    
    # Add spatial reference
    da = da.rio.write_crs('EPSG:4326')
    
    return da


@pytest.fixture  
def sample_temperature_data():
    """Create sample temperature data as xarray Dataset."""
    # Create 1 year of daily data
    time = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    
    # Create spatial grid
    lons = np.arange(-100.25, -98.75, 0.1)
    lats = np.arange(40.25, 41.25, 0.1)
    
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
        
        # Add spatial gradient (cooler to the north)
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
    
    da = da.rio.write_crs('EPSG:4326')
    return da


class TestUtilities:
    """Test utility functions."""
    
    def test_convert_units_precipitation(self):
        """Test precipitation unit conversion."""
        # kg/m²/s to mm/day
        data = np.array([1e-6, 2e-6, 5e-6])
        converted = convert_units(data, "kg/m2/s", "mm/day")
        expected = data * 86400  # seconds per day
        np.testing.assert_array_almost_equal(converted, expected)
    
    def test_convert_units_temperature(self):
        """Test temperature unit conversion."""
        # Kelvin to Celsius
        data = np.array([273.15, 283.15, 293.15])
        converted = convert_units(data, "K", "C")
        expected = np.array([0.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(converted, expected)
    
    def test_get_time_information(self, sample_precipitation_data):
        """Test time information extraction."""
        years, unique_years = get_time_information(sample_precipitation_data)
        
        assert len(years) == len(sample_precipitation_data.time)
        assert set(unique_years) == {2020, 2021}
        assert len(unique_years) == 2
    
    def test_calculate_precipitation_stats(self):
        """Test precipitation statistics calculation."""
        # Create test data
        daily_values = np.array([0.1, 5.0, 15.0, 30.0, 2.0])
        threshold_mm = 25.4
        county_info = {
            'county_id': '12345',
            'county_name': 'Test County',
            'state': 'TX'
        }
        
        stats = calculate_precipitation_stats(
            daily_values, threshold_mm, 2020, 'historical', county_info
        )
        
        assert stats['year'] == 2020
        assert stats['scenario'] == 'historical'
        assert stats['county_id'] == '12345'
        assert stats['total_annual_precip_mm'] == pytest.approx(52.1)
        assert stats['days_above_threshold'] == 1  # Only 30.0 > 25.4
        assert stats['mean_daily_precip_mm'] == pytest.approx(10.42)
        assert stats['max_daily_precip_mm'] == 30.0
        assert stats['dry_days'] == 0  # All values >= 0.1
    
    def test_calculate_temperature_stats(self):
        """Test temperature statistics calculation."""
        # Create test data (in Celsius)
        daily_values = np.array([-5.0, 0.0, 10.0, 25.0, 35.0])
        county_info = {
            'county_id': '12345',
            'county_name': 'Test County',
            'state': 'TX'
        }
        
        stats = calculate_temperature_stats(
            daily_values, 2020, 'historical', county_info
        )
        
        assert stats['year'] == 2020
        assert stats['mean_annual_temp_c'] == pytest.approx(13.0)
        assert stats['min_temp_c'] == -5.0
        assert stats['max_temp_c'] == 35.0
        assert stats['days_below_freezing'] == 1  # Only -5.0 < 0
        assert stats['days_above_30c'] == 1  # Only 35.0 > 30


class TestProcessingStrategies:
    """Test processing strategies."""
    
    def test_vectorized_strategy(self, sample_counties, sample_precipitation_data):
        """Test vectorized processing strategy."""
        strategy = VectorizedStrategy()
        
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
        
        # Should have data for both years
        assert set(results['year'].unique()) == {2020, 2021}
        
        # Should have data for multiple counties
        assert len(results['county_id'].unique()) > 1
    


class TestVariableProcessors:
    """Test variable-specific processors."""
    
    def test_precipitation_processor(self, sample_counties, sample_precipitation_data):
        """Test precipitation processor."""
        processor = PrecipitationProcessor(n_workers=2)
        
        # Prepare counties
        gdf = processor.prepare_shapefile_from_gdf(sample_counties)
        
        # Process data
        results = processor.process_variable_data(
            data=sample_precipitation_data,
            gdf=gdf,
            scenario='test',
            threshold_mm=25.4,
            chunk_by_county=False  # Use vectorized for speed
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        
        # Check precipitation-specific columns
        precip_cols = ['total_annual_precip_mm', 'days_above_threshold', 'dry_days']
        for col in precip_cols:
            assert col in results.columns
    
    def test_temperature_processor(self, sample_counties, sample_temperature_data):
        """Test temperature processor."""
        processor = TemperatureProcessor(n_workers=2)
        
        # Prepare counties
        gdf = processor.prepare_shapefile_from_gdf(sample_counties)
        
        # Process data
        results = processor.process_variable_data(
            data=sample_temperature_data,
            gdf=gdf,
            scenario='test',
            chunk_by_county=False
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        
        # Check temperature-specific columns
        temp_cols = ['mean_annual_temp_c', 'days_below_freezing', 'growing_degree_days']
        for col in temp_cols:
            assert col in results.columns
    
    def test_processor_uses_vectorized_strategy(self, sample_counties, sample_precipitation_data):
        """Test that processors use vectorized strategy."""
        processor = PrecipitationProcessor(n_workers=2)
        
        # Process data - should use VectorizedStrategy internally
        results = processor.process_variable_data(
            data=sample_precipitation_data,
            gdf=sample_counties,
            scenario='test',
            threshold_mm=25.4
        )
        
        # Should produce valid results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0


class TestModernCountyProcessor:
    """Test the unified ModernCountyProcessor interface."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = ModernCountyProcessor(n_workers=4)
        
        assert processor.n_workers == 4
        assert 'pr' in processor._processors
        assert 'tas' in processor._processors
        assert 'tasmax' in processor._processors
        assert 'tasmin' in processor._processors
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with ModernCountyProcessor(n_workers=2) as processor:
            assert processor.n_workers == 2
        # Should not raise any errors on exit
    
    def test_get_processor(self):
        """Test getting specific processors."""
        processor = ModernCountyProcessor()
        
        pr_proc = processor.get_processor('pr')
        assert isinstance(pr_proc, PrecipitationProcessor)
        
        tas_proc = processor.get_processor('tas')
        assert isinstance(tas_proc, TemperatureProcessor)
        
        with pytest.raises(ValueError):
            processor.get_processor('invalid_variable')
    
    def test_prepare_shapefile_from_gdf(self, sample_counties):
        """Test shapefile preparation from GeoDataFrame."""
        processor = ModernCountyProcessor()
        
        # Add method to base processor for testing
        base_proc = processor._processors['pr']
        
        def prepare_shapefile_from_gdf(self, gdf):
            """Helper method for testing."""
            return self._standardize_columns(gdf.copy())
        
        # Monkey patch for testing
        base_proc.prepare_shapefile_from_gdf = prepare_shapefile_from_gdf.__get__(base_proc)
        
        prepared = base_proc.prepare_shapefile_from_gdf(sample_counties)
        
        # Check standardized columns
        required_cols = ['county_id', 'county_name', 'state', 'raster_id', 'geometry']
        for col in required_cols:
            assert col in prepared.columns
        
        assert len(prepared) == len(sample_counties)
        assert prepared['county_id'].iloc[0] == '00000'  # From GEOID


class TestBackwardCompatibility:
    """Test that the public API remains unchanged."""
    
    def test_import_compatibility(self):
        """Test that imports work as before."""
        from climate_zarr import ModernCountyProcessor
        
        # Should be able to create processor
        processor = ModernCountyProcessor()
        assert processor is not None
    
    def test_method_signatures(self):
        """Test that method signatures are compatible."""
        processor = ModernCountyProcessor()
        
        # Check that key methods exist
        assert hasattr(processor, 'prepare_shapefile')
        assert hasattr(processor, 'process_zarr_data')
        assert hasattr(processor, 'close')
        
        # Check context manager methods
        assert hasattr(processor, '__enter__')
        assert hasattr(processor, '__exit__')


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_full_precipitation_workflow(self, sample_counties, sample_precipitation_data):
        """Test complete precipitation processing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save test data as zarr
            zarr_path = Path(tmpdir) / "test_precip.zarr"
            ds = sample_precipitation_data.to_dataset()
            ds.to_zarr(zarr_path)
            
            # Process with ModernCountyProcessor
            with ModernCountyProcessor(n_workers=2) as processor:
                # Prepare counties (simulate from shapefile)
                gdf = processor._processors['pr']._standardize_columns(sample_counties.copy())
                
                # Process zarr data
                results = processor.process_zarr_data(
                    zarr_path=zarr_path,
                    gdf=gdf,
                    scenario='test_scenario',
                    variable='pr',
                    threshold=25.4,
                    chunk_by_county=False
                )
                
                # Validate results
                assert isinstance(results, pd.DataFrame)
                assert len(results) > 0
                assert 'total_annual_precip_mm' in results.columns
                assert set(results['year'].unique()) == {2020, 2021}
                assert results['scenario'].iloc[0] == 'test_scenario'
    
    def test_multiple_variables_workflow(self, sample_counties, sample_precipitation_data, sample_temperature_data):
        """Test processing multiple variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save test data
            precip_path = Path(tmpdir) / "precip.zarr"
            temp_path = Path(tmpdir) / "temp.zarr"
            
            sample_precipitation_data.to_dataset().to_zarr(precip_path)
            sample_temperature_data.to_dataset().to_zarr(temp_path)
            
            with ModernCountyProcessor(n_workers=2) as processor:
                # Prepare counties
                gdf = processor._processors['pr']._standardize_columns(sample_counties.copy())
                
                # Process precipitation
                precip_results = processor.process_zarr_data(
                    zarr_path=precip_path,
                    gdf=gdf,
                    scenario='test',
                    variable='pr',
                    threshold=25.4
                )
                
                # Process temperature
                temp_results = processor.process_zarr_data(
                    zarr_path=temp_path,
                    gdf=gdf,
                    scenario='test',
                    variable='tas'
                )
                
                # Validate both results
                assert len(precip_results) > 0
                assert len(temp_results) > 0
                
                # Should have same counties
                assert set(precip_results['county_id']) == set(temp_results['county_id'])
                
                # Different column sets
                assert 'total_annual_precip_mm' in precip_results.columns
                assert 'mean_annual_temp_c' in temp_results.columns
                assert 'total_annual_precip_mm' not in temp_results.columns
                assert 'mean_annual_temp_c' not in precip_results.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 