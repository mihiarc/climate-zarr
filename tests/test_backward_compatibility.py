#!/usr/bin/env python
"""Tests to ensure backward compatibility after modular refactoring."""

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
import inspect
from unittest.mock import patch, Mock

from climate_zarr import ModernCountyProcessor


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_shapefile(temp_dir):
    """Create a sample shapefile for testing."""
    counties = []
    for i in range(3):
        poly = Polygon([(-100, 40), (-99, 40), (-99, 41), (-100, 41)])
        counties.append({
            'GEOID': f'{i:05d}',
            'NAME': f'Test County {i}',
            'STUSPS': 'TX',
            'geometry': poly
        })
    
    gdf = gpd.GeoDataFrame(counties, crs='EPSG:4326')
    shapefile_path = Path(temp_dir) / "test_counties.shp"
    gdf.to_file(shapefile_path)
    
    return shapefile_path


@pytest.fixture
def sample_zarr_data(temp_dir):
    """Create sample zarr data for testing."""
    time = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    lons = np.arange(-100.1, -98.9, 0.1)
    lats = np.arange(40.1, 40.9, 0.1)
    
    data = np.random.exponential(2e-6, size=(len(time), len(lats), len(lons)))
    
    da = xr.DataArray(
        data,
        coords={'time': time, 'lat': lats, 'lon': lons},
        dims=['time', 'lat', 'lon'],
        name='pr'
    )
    
    da = da.rio.write_crs('EPSG:4326')
    da.attrs['units'] = 'kg/m2/s'
    
    zarr_path = Path(temp_dir) / "test_data.zarr"
    ds = da.to_dataset()
    ds.to_zarr(zarr_path)
    
    return zarr_path


class TestImportCompatibility:
    """Test that all expected imports still work."""
    
    def test_main_import(self):
        """Test that the main ModernCountyProcessor import works."""
        from climate_zarr import ModernCountyProcessor
        
        assert ModernCountyProcessor is not None
        
        # Should be able to instantiate
        processor = ModernCountyProcessor()
        assert processor is not None
    
    def test_legacy_imports(self):
        """Test that legacy import patterns still work."""
        # Test importing from main module
        from climate_zarr import ModernCountyProcessor
        
        # Should be able to create instance
        processor = ModernCountyProcessor(n_workers=2)
        assert processor.n_workers == 2
    
    def test_new_modular_imports(self):
        """Test that new modular imports work."""
        # Test importing individual processors
        from climate_zarr.processors import (
            PrecipitationProcessor,
            TemperatureProcessor,
            TasMaxProcessor,
            TasMinProcessor
        )
        
        # Should be able to create instances
        precip_proc = PrecipitationProcessor()
        temp_proc = TemperatureProcessor()
        tasmax_proc = TasMaxProcessor()
        tasmin_proc = TasMinProcessor()
        
        assert precip_proc is not None
        assert temp_proc is not None
        assert tasmax_proc is not None
        assert tasmin_proc is not None
        
        # Test importing strategies
        from climate_zarr.processors.processing_strategies import VectorizedStrategy
        
        vectorized = VectorizedStrategy()
        
        assert vectorized is not None
    
    def test_utility_imports(self):
        """Test that utility imports work."""
        from climate_zarr.utils import (
            convert_units,
            create_county_raster,
            get_time_information
        )
        
        assert convert_units is not None
        assert create_county_raster is not None
        assert get_time_information is not None
        
        # Test data utils
        from climate_zarr.utils.data_utils import (
            calculate_precipitation_stats,
            calculate_temperature_stats
        )
        
        assert calculate_precipitation_stats is not None
        assert calculate_temperature_stats is not None


class TestAPICompatibility:
    """Test that the public API remains compatible."""
    
    def test_processor_initialization(self):
        """Test that processor initialization works as before."""
        # Test with default parameters
        processor = ModernCountyProcessor()
        assert processor is not None
        
        # Test with custom parameters
        processor = ModernCountyProcessor(n_workers=4)
        assert processor.n_workers == 4
    
    def test_context_manager_compatibility(self):
        """Test that context manager functionality still works."""
        # Test entering and exiting context
        with ModernCountyProcessor(n_workers=2) as processor:
            assert processor is not None
            assert processor.n_workers == 2
        
        # Should not raise any errors on exit
    
    def test_method_existence(self):
        """Test that all expected methods exist."""
        processor = ModernCountyProcessor()
        
        # Core methods should exist
        assert hasattr(processor, 'prepare_shapefile')
        assert hasattr(processor, 'process_zarr_data')
        assert hasattr(processor, 'close')
        
        # Context manager methods should exist
        assert hasattr(processor, '__enter__')
        assert hasattr(processor, '__exit__')
        
        # New methods should exist
        assert hasattr(processor, 'get_processor')
    
    def test_method_signatures(self):
        """Test that method signatures remain compatible."""
        processor = ModernCountyProcessor()
        
        # Test prepare_shapefile signature
        sig = inspect.signature(processor.prepare_shapefile)
        assert 'shapefile_path' in sig.parameters
        
        # Test process_zarr_data signature
        sig = inspect.signature(processor.process_zarr_data)
        expected_params = ['zarr_path', 'gdf', 'scenario', 'variable']
        for param in expected_params:
            assert param in sig.parameters
        
        # Optional parameters should have defaults
        assert sig.parameters['threshold'].default == 25.4  # Default precipitation threshold
        assert sig.parameters['chunk_by_county'].default is True
    
    def test_prepare_shapefile_compatibility(self, sample_shapefile):
        """Test that prepare_shapefile works as before."""
        processor = ModernCountyProcessor()
        
        # Should accept string path
        gdf = processor.prepare_shapefile(str(sample_shapefile))
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0
        
        # Should accept Path object
        gdf = processor.prepare_shapefile(sample_shapefile)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0
        
        # Should have standardized columns
        required_cols = ['county_id', 'county_name', 'state', 'geometry']
        for col in required_cols:
            assert col in gdf.columns
    
    def test_process_zarr_data_compatibility(self, sample_shapefile, sample_zarr_data):
        """Test that process_zarr_data works as before."""
        processor = ModernCountyProcessor()
        
        # Prepare shapefile
        gdf = processor.prepare_shapefile(sample_shapefile)
        
        # Process data with all parameters
        results = processor.process_zarr_data(
            zarr_path=sample_zarr_data,
            gdf=gdf,
            scenario='test_scenario',
            variable='pr',
            threshold=25.4,
            chunk_by_county=True
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        
        # Check that all expected columns are present
        required_cols = ['year', 'scenario', 'county_id', 'county_name', 'state']
        for col in required_cols:
            assert col in results.columns
        
        # Check that scenario is preserved
        assert results['scenario'].iloc[0] == 'test_scenario'
        
        # Test with minimal parameters
        results = processor.process_zarr_data(
            zarr_path=sample_zarr_data,
            gdf=gdf,
            scenario='minimal_test',
            variable='pr'
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
    
    def test_close_compatibility(self):
        """Test that close method works as before."""
        processor = ModernCountyProcessor()
        
        # Should not raise any errors
        processor.close()
        
        # Should be able to call multiple times
        processor.close()


class TestDataCompatibility:
    """Test that data processing produces compatible results."""
    
    def test_precipitation_results_structure(self, sample_shapefile, sample_zarr_data):
        """Test that precipitation results have expected structure."""
        processor = ModernCountyProcessor()
        gdf = processor.prepare_shapefile(sample_shapefile)
        
        results = processor.process_zarr_data(
            zarr_path=sample_zarr_data,
            gdf=gdf,
            scenario='test',
            variable='pr',
            threshold=25.4
        )
        
        # Check expected precipitation columns
        expected_cols = [
            'year', 'scenario', 'county_id', 'county_name', 'state',
            'total_annual_precip_mm', 'days_above_threshold', 'dry_days',
            'mean_daily_precip_mm', 'max_daily_precip_mm'
        ]
        
        for col in expected_cols:
            assert col in results.columns
        
        # Check data types
        assert results['year'].dtype in [np.int32, np.int64]
        assert results['scenario'].dtype == object
        assert results['county_id'].dtype == object
        assert results['total_annual_precip_mm'].dtype in [np.float32, np.float64]
        assert results['days_above_threshold'].dtype in [np.int32, np.int64]
    
    def test_temperature_results_structure(self, sample_shapefile, temp_dir):
        """Test that temperature results have expected structure."""
        # Create temperature data
        time = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        lons = np.arange(-100.1, -98.9, 0.1)
        lats = np.arange(40.1, 40.9, 0.1)
        
        data = np.random.normal(283.15, 10, size=(len(time), len(lats), len(lons)))
        
        da = xr.DataArray(
            data,
            coords={'time': time, 'lat': lats, 'lon': lons},
            dims=['time', 'lat', 'lon'],
            name='tas'
        )
        
        da = da.rio.write_crs('EPSG:4326')
        da.attrs['units'] = 'K'
        
        zarr_path = Path(temp_dir) / "temp_data.zarr"
        ds = da.to_dataset()
        ds.to_zarr(zarr_path)
        
        # Process temperature data
        processor = ModernCountyProcessor()
        gdf = processor.prepare_shapefile(sample_shapefile)
        
        results = processor.process_zarr_data(
            zarr_path=zarr_path,
            gdf=gdf,
            scenario='test',
            variable='tas'
        )
        
        # Check expected temperature columns
        expected_cols = [
            'year', 'scenario', 'county_id', 'county_name', 'state',
            'mean_annual_temp_c', 'days_below_freezing', 'growing_degree_days',
            'min_temp_c', 'max_temp_c', 'days_above_30c'
        ]
        
        for col in expected_cols:
            assert col in results.columns
        
        # Check data types
        assert results['mean_annual_temp_c'].dtype in [np.float32, np.float64]
        assert results['days_below_freezing'].dtype in [np.int32, np.int64]
        assert results['growing_degree_days'].dtype in [np.int32, np.int64, np.float64]
    
    def test_data_value_ranges(self, sample_shapefile, sample_zarr_data):
        """Test that data values are in expected ranges."""
        processor = ModernCountyProcessor()
        gdf = processor.prepare_shapefile(sample_shapefile)
        
        results = processor.process_zarr_data(
            zarr_path=sample_zarr_data,
            gdf=gdf,
            scenario='test',
            variable='pr',
            threshold=25.4
        )
        
        # Precipitation values should be reasonable
        assert results['total_annual_precip_mm'].min() >= 0
        assert results['total_annual_precip_mm'].max() < 10000  # Not too high
        assert results['days_above_threshold'].min() >= 0
        assert results['days_above_threshold'].max() <= 365
        assert results['dry_days'].min() >= 0
        assert results['dry_days'].max() <= 365
    
    def test_county_id_consistency(self, sample_shapefile, sample_zarr_data):
        """Test that county IDs are consistent with input shapefile."""
        processor = ModernCountyProcessor()
        gdf = processor.prepare_shapefile(sample_shapefile)
        
        results = processor.process_zarr_data(
            zarr_path=sample_zarr_data,
            gdf=gdf,
            scenario='test',
            variable='pr'
        )
        
        # County IDs should match those in the prepared shapefile
        result_county_ids = set(results['county_id'].unique())
        shapefile_county_ids = set(gdf['county_id'].unique())
        
        assert result_county_ids.issubset(shapefile_county_ids)
        
        # Should have results for all counties (or at least some)
        assert len(result_county_ids) > 0


class TestErrorHandlingCompatibility:
    """Test that error handling remains compatible."""
    
    def test_file_not_found_errors(self):
        """Test that file not found errors are handled consistently."""
        processor = ModernCountyProcessor()
        
        # Test with non-existent shapefile (pyogrio raises DataSourceError)
        with pytest.raises((FileNotFoundError, Exception)):  # Accept various file-not-found errors
            processor.prepare_shapefile("nonexistent.shp")
        
        # Test with non-existent zarr file
        gdf = gpd.GeoDataFrame({
            'county_id': ['12345'],
            'county_name': ['Test'],
            'state': ['TX'],
            'geometry': [Polygon([(-100, 40), (-99, 40), (-99, 41), (-100, 41)])]
        })
        
        with pytest.raises(FileNotFoundError):
            processor.process_zarr_data(
                zarr_path="nonexistent.zarr",
                gdf=gdf,
                scenario='test',
                variable='pr'
            )
    
    def test_invalid_variable_errors(self, sample_shapefile, sample_zarr_data):
        """Test that invalid variable errors are handled consistently."""
        processor = ModernCountyProcessor()
        gdf = processor.prepare_shapefile(sample_shapefile)
        
        with pytest.raises(ValueError, match="Unsupported variable"):
            processor.process_zarr_data(
                zarr_path=sample_zarr_data,
                gdf=gdf,
                scenario='test',
                variable='invalid_variable'
            )
    
    def test_empty_geodataframe_handling(self, sample_zarr_data):
        """Test handling of empty GeoDataFrame."""
        processor = ModernCountyProcessor()
        
        # Create empty GeoDataFrame
        empty_gdf = gpd.GeoDataFrame(columns=['county_id', 'county_name', 'state', 'geometry'])
        
        # Should handle empty GeoDataFrame gracefully
        results = processor.process_zarr_data(
            zarr_path=sample_zarr_data,
            gdf=empty_gdf,
            scenario='test',
            variable='pr'
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0


class TestPerformanceCompatibility:
    """Test that performance characteristics remain compatible."""
    
    def test_memory_usage_reasonable(self, sample_shapefile, sample_zarr_data):
        """Test that memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        processor = ModernCountyProcessor()
        gdf = processor.prepare_shapefile(sample_shapefile)
        
        results = processor.process_zarr_data(
            zarr_path=sample_zarr_data,
            gdf=gdf,
            scenario='test',
            variable='pr'
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 200MB for test data)
        assert memory_increase < 200, f"Memory usage increased by {memory_increase:.1f}MB"
        
        # Should still produce valid results
        assert len(results) > 0
    
    def test_processing_time_reasonable(self, sample_shapefile, sample_zarr_data):
        """Test that processing time is reasonable."""
        import time
        
        processor = ModernCountyProcessor()
        gdf = processor.prepare_shapefile(sample_shapefile)
        
        start_time = time.time()
        
        results = processor.process_zarr_data(
            zarr_path=sample_zarr_data,
            gdf=gdf,
            scenario='test',
            variable='pr'
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (10 seconds for test data)
        assert processing_time < 10, f"Processing took {processing_time:.1f} seconds"
        
        # Should produce valid results
        assert len(results) > 0


class TestConfigurationCompatibility:
    """Test that configuration options remain compatible."""
    
    def test_n_workers_parameter(self, sample_shapefile, sample_zarr_data):
        """Test that n_workers parameter works as before."""
        # Test with different worker counts
        for n_workers in [1, 2, 4]:
            processor = ModernCountyProcessor(n_workers=n_workers)
            assert processor.n_workers == n_workers
            
            gdf = processor.prepare_shapefile(sample_shapefile)
            
            results = processor.process_zarr_data(
                zarr_path=sample_zarr_data,
                gdf=gdf,
                scenario='test',
                variable='pr'
            )
            
            assert len(results) > 0
    
    def test_threshold_parameter(self, sample_shapefile, sample_zarr_data):
        """Test that threshold parameter works as before."""
        processor = ModernCountyProcessor()
        gdf = processor.prepare_shapefile(sample_shapefile)
        
        # Test with different thresholds
        for threshold in [10.0, 25.4, 50.0]:
            results = processor.process_zarr_data(
                zarr_path=sample_zarr_data,
                gdf=gdf,
                scenario='test',
                variable='pr',
                threshold=threshold
            )
            
            assert len(results) > 0
            assert 'days_above_threshold' in results.columns
    
    def test_chunk_by_county_parameter(self, sample_shapefile, sample_zarr_data):
        """Test that chunk_by_county parameter works as before."""
        processor = ModernCountyProcessor()
        gdf = processor.prepare_shapefile(sample_shapefile)
        
        # Test with chunk_by_county=True
        results_chunked = processor.process_zarr_data(
            zarr_path=sample_zarr_data,
            gdf=gdf,
            scenario='test_chunked',
            variable='pr',
            chunk_by_county=True
        )
        
        # Test with chunk_by_county=False
        results_not_chunked = processor.process_zarr_data(
            zarr_path=sample_zarr_data,
            gdf=gdf,
            scenario='test_not_chunked',
            variable='pr',
            chunk_by_county=False
        )
        
        # Both should produce results
        assert len(results_chunked) > 0
        assert len(results_not_chunked) > 0
        
        # Should have same structure
        assert list(results_chunked.columns) == list(results_not_chunked.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 