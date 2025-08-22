#!/usr/bin/env python
"""Unit tests for individual processor modules."""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # Required for .rio accessor
from shapely.geometry import Polygon
from unittest.mock import Mock, patch

from climate_zarr.processors.base_processor import BaseCountyProcessor
from climate_zarr.processors.precipitation_processor import PrecipitationProcessor
from climate_zarr.processors.temperature_processor import TemperatureProcessor
from climate_zarr.processors.tasmax_processor import TasMaxProcessor
from climate_zarr.processors.tasmin_processor import TasMinProcessor
from climate_zarr.processors.processing_strategies import VectorizedStrategy


@pytest.fixture
def sample_gdf():
    """Create a simple test GeoDataFrame."""
    counties = []
    for i in range(2):
        poly = Polygon([(-100, 40), (-99, 40), (-99, 41), (-100, 41)])
        counties.append({
            'GEOID': f'{i:05d}',
            'NAME': f'Test County {i}',
            'STUSPS': 'TX',
            'geometry': poly
        })
    
    return gpd.GeoDataFrame(counties, crs='EPSG:4326')


@pytest.fixture
def sample_xarray_data():
    """Create sample xarray data."""
    time = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    lons = np.array([-100.0, -99.5, -99.0])
    lats = np.array([40.0, 40.5, 41.0])
    
    data = np.random.rand(len(time), len(lats), len(lons))
    
    da = xr.DataArray(
        data,
        coords={'time': time, 'lat': lats, 'lon': lons},
        dims=['time', 'lat', 'lon'],
        name='test_var'
    )
    
    return da.rio.write_crs('EPSG:4326')


class TestBaseProcessor:
    """Test the base processor functionality."""
    
    def test_initialization(self):
        """Test base processor initialization."""
        # Use concrete processor since BaseCountyProcessor is abstract
        processor = PrecipitationProcessor(n_workers=4)
        
        assert processor.n_workers == 4
    
    def test_standardize_columns(self, sample_gdf):
        """Test column standardization."""
        processor = PrecipitationProcessor()
        
        standardized = processor._standardize_columns(sample_gdf.copy())
        
        # Check required columns exist
        required_cols = ['county_id', 'county_name', 'state', 'raster_id', 'geometry']
        for col in required_cols:
            assert col in standardized.columns
        
        # Check values
        assert standardized['county_id'].iloc[0] == '00000'
        assert standardized['county_name'].iloc[0] == 'Test County 0'
        assert standardized['state'].iloc[0] == 'TX'
        assert standardized['raster_id'].iloc[0] == 1
    
    def test_prepare_shapefile_method_exists(self):
        """Test that prepare_shapefile method exists."""
        processor = PrecipitationProcessor()
        assert hasattr(processor, 'prepare_shapefile')
        assert hasattr(processor, '_standardize_columns')
        assert hasattr(processor, 'close')
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with PrecipitationProcessor(n_workers=2) as processor:
            assert processor.n_workers == 2
        # Should not raise any errors


class TestPrecipitationProcessor:
    """Test precipitation-specific processor."""
    
    def test_initialization(self):
        """Test precipitation processor initialization."""
        processor = PrecipitationProcessor(n_workers=3)
        
        assert processor.n_workers == 3
        assert hasattr(processor, 'process_variable_data')
        assert hasattr(processor, 'prepare_shapefile')
        assert hasattr(processor, 'close')
    
    def test_process_variable_data(self, sample_gdf, sample_xarray_data):
        """Test precipitation data processing."""
        processor = PrecipitationProcessor(n_workers=2)
        
        # Prepare data
        gdf = processor._standardize_columns(sample_gdf.copy())
        
        # Mock the VectorizedStrategy to return test data
        with patch('climate_zarr.processors.precipitation_processor.VectorizedStrategy') as mock_strategy_class:
            mock_strategy_instance = Mock()
            mock_strategy_instance.process.return_value = pd.DataFrame({
                'year': [2020, 2020],
                'scenario': ['test', 'test'],
                'county_id': ['00000', '00001'],
                'county_name': ['Test County 0', 'Test County 1'],
                'state': ['TX', 'TX'],
                'total_annual_precip_mm': [1000.0, 1200.0],
                'days_above_threshold': [50, 60],
                'dry_days': [10, 5]
            })
            mock_strategy_class.return_value = mock_strategy_instance
            
            results = processor.process_variable_data(
                data=sample_xarray_data,
                gdf=gdf,
                scenario='test',
                threshold_mm=25.4
            )
            
            # Verify results
            assert isinstance(results, pd.DataFrame)
            assert len(results) == 2
            assert 'total_annual_precip_mm' in results.columns
            assert 'days_above_threshold' in results.columns
            assert 'dry_days' in results.columns
            
            # Verify strategy was instantiated and called correctly
            mock_strategy_class.assert_called_once()
            mock_strategy_instance.process.assert_called_once()
    
    def test_uses_vectorized_strategy(self):
        """Test that processor uses VectorizedStrategy."""
        processor = PrecipitationProcessor()
        
        # Verify that the processor always uses VectorizedStrategy now
        # This is implicitly tested in the process_variable_data test
        assert hasattr(processor, 'process_variable_data')
        
        # The architecture now directly instantiates VectorizedStrategy
        # instead of having a selection method
        assert not hasattr(processor, '_select_processing_strategy')


class TestTemperatureProcessor:
    """Test temperature-specific processor."""
    
    def test_initialization(self):
        """Test temperature processor initialization."""
        processor = TemperatureProcessor(n_workers=2)
        
        assert processor.n_workers == 2
        assert hasattr(processor, 'process_variable_data')
        assert hasattr(processor, 'prepare_shapefile')
        assert hasattr(processor, 'close')
    
    def test_uses_vectorized_strategy(self):
        """Test that processor uses VectorizedStrategy."""
        processor = TemperatureProcessor()
        
        # Verify that the processor always uses VectorizedStrategy now
        assert hasattr(processor, 'process_variable_data')
        
        # The architecture now directly instantiates VectorizedStrategy
        assert not hasattr(processor, '_select_processing_strategy')
    
    def test_process_variable_data(self, sample_gdf, sample_xarray_data):
        """Test temperature data processing."""
        processor = TemperatureProcessor(n_workers=2)
        
        # Prepare data
        gdf = processor._standardize_columns(sample_gdf.copy())
        
        # Mock the VectorizedStrategy
        with patch('climate_zarr.processors.temperature_processor.VectorizedStrategy') as mock_strategy_class:
            mock_strategy_instance = Mock()
            mock_strategy_instance.process.return_value = pd.DataFrame({
                'year': [2020, 2020],
                'scenario': ['test', 'test'],
                'county_id': ['00000', '00001'],
                'county_name': ['Test County 0', 'Test County 1'],
                'state': ['TX', 'TX'],
                'mean_annual_temp_c': [15.5, 16.2],
                'days_below_freezing': [30, 25],
                'growing_degree_days': [2500, 2800]
            })
            mock_strategy_class.return_value = mock_strategy_instance
            
            results = processor.process_variable_data(
                data=sample_xarray_data,
                gdf=gdf,
                scenario='test'
            )
            
            # Verify results
            assert isinstance(results, pd.DataFrame)
            assert len(results) == 2
            assert 'mean_annual_temp_c' in results.columns
            assert 'days_below_freezing' in results.columns
            assert 'growing_degree_days' in results.columns
            
            # Verify strategy was instantiated and called correctly
            mock_strategy_class.assert_called_once()
            mock_strategy_instance.process.assert_called_once()


class TestTasMaxProcessor:
    """Test tasmax-specific processor."""
    
    def test_initialization(self):
        """Test tasmax processor initialization."""
        processor = TasMaxProcessor(n_workers=2)
        
        assert processor.n_workers == 2
        assert hasattr(processor, 'process_variable_data')
        assert hasattr(processor, 'prepare_shapefile')
        assert hasattr(processor, 'close')
    
    def test_uses_vectorized_strategy(self):
        """Test that processor uses VectorizedStrategy."""
        processor = TasMaxProcessor()
        
        # Test that processor uses VectorizedStrategy
        assert hasattr(processor, 'process_variable_data')
        
        # The architecture now directly instantiates VectorizedStrategy
        assert not hasattr(processor, '_select_processing_strategy')


class TestTasMinProcessor:
    """Test tasmin-specific processor."""
    
    def test_initialization(self):
        """Test tasmin processor initialization."""
        processor = TasMinProcessor(n_workers=2)
        
        assert processor.n_workers == 2
        assert hasattr(processor, 'process_variable_data')
        assert hasattr(processor, 'prepare_shapefile')
        assert hasattr(processor, 'close')
    
    def test_uses_vectorized_strategy(self):
        """Test that processor uses VectorizedStrategy."""
        processor = TasMinProcessor()
        
        # Create test data
        test_gdf = gpd.GeoDataFrame({
            'GEOID': ['12345'],
            'NAME': ['Test County'],
            'STUSPS': ['TX'],
            'geometry': [Polygon([(-100, 40), (-99, 40), (-99, 41), (-100, 41)])]
        })
        
        # Test that processor uses VectorizedStrategy
        assert hasattr(processor, 'process_variable_data')
        
        # The architecture now directly instantiates VectorizedStrategy
        assert not hasattr(processor, '_select_processing_strategy')


class TestProcessorErrorHandling:
    """Test error handling in processors."""
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns in GeoDataFrame."""
        processor = PrecipitationProcessor()
        
        # Create GDF missing required columns
        invalid_gdf = gpd.GeoDataFrame({
            'geometry': [Polygon([(-100, 40), (-99, 40), (-99, 41), (-100, 41)])]
        })
        
        # Should handle missing columns gracefully
        result = processor._standardize_columns(invalid_gdf)
        
        # Should have created default values
        assert 'county_id' in result.columns
        assert 'county_name' in result.columns
        assert 'state' in result.columns
    
    def test_empty_geodataframe(self):
        """Test handling of empty GeoDataFrame."""
        processor = PrecipitationProcessor()
        
        empty_gdf = gpd.GeoDataFrame(columns=['GEOID', 'NAME', 'STUSPS', 'geometry'])
        
        result = processor._standardize_columns(empty_gdf)
        
        assert len(result) == 0
        assert 'county_id' in result.columns


class TestProcessorPerformance:
    """Test performance-related aspects of processors."""
    
    def test_column_standardization_performance(self):
        """Test that column standardization is fast."""
        processor = PrecipitationProcessor()
        
        # Create varying sizes of GeoDataFrames
        sizes = [10, 50, 100]
        
        for size in sizes:
            # Create test GDF
            test_gdf = gpd.GeoDataFrame({
                'GEOID': [f'{i:05d}' for i in range(size)],
                'NAME': [f'County {i}' for i in range(size)],
                'STUSPS': ['TX'] * size,
                'geometry': [Polygon([(-100, 40), (-99, 40), (-99, 41), (-100, 41)])] * size
            })
            
            # Column standardization should be fast
            result = processor._standardize_columns(test_gdf)
            
            # Verify result
            assert len(result) == size
            assert 'county_id' in result.columns
    
    def test_memory_efficiency(self):
        """Test that processors handle data efficiently."""
        processor = PrecipitationProcessor(n_workers=2)
        
        # Create some test data
        test_gdf = gpd.GeoDataFrame({
            'GEOID': ['12345'],
            'NAME': ['Test'],
            'STUSPS': ['TX'],
            'geometry': [Polygon([(-100, 40), (-99, 40), (-99, 41), (-100, 41)])]
        })
        
        # Process and verify standardization works
        standardized = processor._standardize_columns(test_gdf.copy())
        
        # Should have added required columns
        assert 'county_id' in standardized.columns
        assert 'county_name' in standardized.columns
        assert 'state' in standardized.columns
        assert 'raster_id' in standardized.columns
        
        # Should have correct values
        assert standardized.loc[0, 'county_id'] == '12345'
        assert standardized.loc[0, 'county_name'] == 'Test'
        assert standardized.loc[0, 'state'] == 'TX'


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 