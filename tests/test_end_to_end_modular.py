#!/usr/bin/env python
"""End-to-end integration tests for the modular climate-zarr system."""

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
import os

from climate_zarr.county_processor import ModernCountyProcessor
from climate_zarr import county_stats_cli


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_counties_shapefile(temp_dir):
    """Create a sample counties shapefile."""
    counties = []
    for i in range(5):
        # Create rectangular counties
        minx, miny = -100 + i*0.3, 40
        maxx, maxy = minx + 0.3, 41
        
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
    
    gdf = gpd.GeoDataFrame(counties, crs='EPSG:4326')
    
    # Save as shapefile
    shapefile_path = Path(temp_dir) / "test_counties.shp"
    gdf.to_file(shapefile_path)
    
    return shapefile_path


@pytest.fixture
def sample_precipitation_zarr(temp_dir):
    """Create a sample precipitation zarr dataset."""
    # Create 2 years of daily data
    time = pd.date_range('2020-01-01', '2021-12-31', freq='D')
    
    # Create spatial grid covering the counties
    lons = np.arange(-100.2, -98.8, 0.05)  # 28 points
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
        coords={
            'time': time,
            'lat': lats,
            'lon': lons
        },
        dims=['time', 'lat', 'lon'],
        name='pr'
    )
    
    # Add spatial reference and attributes
    da = da.rio.write_crs('EPSG:4326')
    da.attrs['units'] = 'kg/m2/s'
    da.attrs['long_name'] = 'precipitation'
    
    # Save as zarr
    zarr_path = Path(temp_dir) / "test_precipitation.zarr"
    ds = da.to_dataset()
    ds.to_zarr(zarr_path)
    
    return zarr_path


@pytest.fixture
def sample_temperature_zarr(temp_dir):
    """Create a sample temperature zarr dataset."""
    # Create 1 year of daily data
    time = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    
    # Create spatial grid
    lons = np.arange(-100.2, -98.8, 0.05)
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
        coords={
            'time': time,
            'lat': lats,
            'lon': lons
        },
        dims=['time', 'lat', 'lon'],
        name='tas'
    )
    
    # Add spatial reference and attributes
    da = da.rio.write_crs('EPSG:4326')
    da.attrs['units'] = 'K'
    da.attrs['long_name'] = 'air_temperature'
    
    # Save as zarr
    zarr_path = Path(temp_dir) / "test_temperature.zarr"
    ds = da.to_dataset()
    ds.to_zarr(zarr_path)
    
    return zarr_path


class TestModernCountyProcessorIntegration:
    """Test the ModernCountyProcessor with real file I/O."""
    
    def test_full_precipitation_workflow(self, sample_counties_shapefile, sample_precipitation_zarr, temp_dir):
        """Test complete precipitation processing workflow."""
        output_path = Path(temp_dir) / "precipitation_results.csv"
        
        with ModernCountyProcessor(n_workers=2) as processor:
            # Load shapefile
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            # Process zarr data
            results = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                scenario='ssp370',
                variable='pr',
                threshold=25.4,
                chunk_by_county=False  # Use ultra-fast strategy
            )
            
            # Save results
            results.to_csv(output_path, index=False)
            
            # Validate results
            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0
            assert output_path.exists()
            
            # Check required columns
            required_cols = [
                'year', 'scenario', 'county_id', 'county_name', 'state',
                'total_annual_precip_mm', 'days_above_threshold', 'dry_days'
            ]
            for col in required_cols:
                assert col in results.columns
            
            # Should have data for both years
            assert set(results['year'].unique()) == {2020, 2021}
            
            # Should have data for all counties
            assert len(results['county_id'].unique()) == 5
            
            # Values should be reasonable
            assert results['total_annual_precip_mm'].min() >= 0
            assert results['days_above_threshold'].min() >= 0
            assert results['dry_days'].min() >= 0
            
            # Check that file was written correctly
            loaded_results = pd.read_csv(output_path)
            assert len(loaded_results) == len(results)
    
    def test_full_temperature_workflow(self, sample_counties_shapefile, sample_temperature_zarr, temp_dir):
        """Test complete temperature processing workflow."""
        output_path = Path(temp_dir) / "temperature_results.csv"
        
        with ModernCountyProcessor(n_workers=2) as processor:
            # Load shapefile
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            # Process zarr data
            results = processor.process_zarr_data(
                zarr_path=sample_temperature_zarr,
                gdf=gdf,
                scenario='historical',
                variable='tas',
                chunk_by_county=True  # Use vectorized strategy
            )
            
            # Save results
            results.to_csv(output_path, index=False)
            
            # Validate results
            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0
            assert output_path.exists()
            
            # Check temperature-specific columns
            temp_cols = [
                'mean_annual_temp_c', 'days_below_freezing', 'growing_degree_days',
                'min_temp_c', 'max_temp_c', 'days_above_30c'
            ]
            for col in temp_cols:
                assert col in results.columns
            
            # Should have data for 2020
            assert set(results['year'].unique()) == {2020}
            
            # Temperature values should be reasonable
            assert results['mean_annual_temp_c'].min() > -50  # Not too cold
            assert results['mean_annual_temp_c'].max() < 50   # Not too hot
            assert results['days_below_freezing'].min() >= 0
            assert results['growing_degree_days'].min() >= 0
    
    def test_multiple_variables_workflow(self, sample_counties_shapefile, sample_precipitation_zarr, sample_temperature_zarr, temp_dir):
        """Test processing multiple variables in sequence."""
        with ModernCountyProcessor(n_workers=2) as processor:
            # Load shapefile once
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            # Process precipitation
            precip_results = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                scenario='ssp370',
                variable='pr',
                threshold=25.4
            )
            
            # Process temperature
            temp_results = processor.process_zarr_data(
                zarr_path=sample_temperature_zarr,
                gdf=gdf,
                scenario='historical',
                variable='tas'
            )
            
            # Both should have results
            assert len(precip_results) > 0
            assert len(temp_results) > 0
            
            # Should have same counties
            assert set(precip_results['county_id']) == set(temp_results['county_id'])
            
            # Different scenarios should be preserved
            assert precip_results['scenario'].iloc[0] == 'ssp370'
            assert temp_results['scenario'].iloc[0] == 'historical'
            
            # Different column sets
            assert 'total_annual_precip_mm' in precip_results.columns
            assert 'mean_annual_temp_c' in temp_results.columns
            assert 'total_annual_precip_mm' not in temp_results.columns
            assert 'mean_annual_temp_c' not in precip_results.columns
    
    def test_strategy_selection_integration(self, sample_counties_shapefile, sample_precipitation_zarr):
        """Test that strategy selection works correctly in integration."""
        with ModernCountyProcessor(n_workers=2) as processor:
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            # Test with chunk_by_county=True (should use vectorized for small dataset)
            results_vectorized = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                scenario='test_vectorized',
                variable='pr',
                threshold=25.4,
                chunk_by_county=True
            )
            
            # Test with chunk_by_county=False (should use ultra-fast)
            results_ultrafast = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                scenario='test_ultrafast',
                variable='pr',
                threshold=25.4,
                chunk_by_county=False
            )
            
            # Both should produce results
            assert len(results_vectorized) > 0
            assert len(results_ultrafast) > 0
            
            # Should have same structure
            assert len(results_vectorized) == len(results_ultrafast)
            assert list(results_vectorized.columns) == list(results_ultrafast.columns)
            
            # Scenarios should be different
            assert results_vectorized['scenario'].iloc[0] == 'test_vectorized'
            assert results_ultrafast['scenario'].iloc[0] == 'test_ultrafast'
    
    def test_error_handling_integration(self, sample_counties_shapefile, temp_dir):
        """Test error handling in integration scenarios."""
        with ModernCountyProcessor(n_workers=2) as processor:
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            # Test with non-existent zarr file
            with pytest.raises(FileNotFoundError):
                processor.process_zarr_data(
                    zarr_path=Path(temp_dir) / "nonexistent.zarr",
                    gdf=gdf,
                    scenario='test',
                    variable='pr',
                    threshold=25.4
                )
            
            # Test with invalid variable
            with pytest.raises(ValueError):
                processor.process_zarr_data(
                    zarr_path=sample_precipitation_zarr,
                    gdf=gdf,
                    scenario='test',
                    variable='invalid_variable',
                    threshold=25.4
                )


class TestCountyProcessorIntegration:
    """Test the unified ModernCountyProcessor interface."""
    
    def test_county_processor_initialization(self):
        """Test ModernCountyProcessor initialization."""
        processor = ModernCountyProcessor(n_workers=4)
        
        assert processor.n_workers == 4
        assert hasattr(processor, 'prepare_shapefile')
        assert hasattr(processor, 'process_zarr_data')
        assert hasattr(processor, 'get_processor')
        assert hasattr(processor, 'close')
    
    def test_county_processor_precipitation(self, sample_counties_shapefile, sample_precipitation_zarr):
        """Test ModernCountyProcessor precipitation processing."""
        processor = ModernCountyProcessor(n_workers=2)
        
        # Load shapefile
        gdf = processor.prepare_shapefile(sample_counties_shapefile)
        
        # Process precipitation data
        results = processor.process_zarr_data(
            zarr_path=sample_precipitation_zarr,
            gdf=gdf,
            scenario='ssp370',
            variable='pr',
            threshold=25.4,
            chunk_by_county=False
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'total_annual_precip_mm' in results.columns
        assert set(results['scenario'].unique()) == {'ssp370'}
    
    def test_county_processor_temperature(self, sample_counties_shapefile, sample_temperature_zarr):
        """Test ModernCountyProcessor temperature processing."""
        processor = ModernCountyProcessor(n_workers=2)
        
        # Load shapefile
        gdf = processor.prepare_shapefile(sample_counties_shapefile)
        
        # Process temperature data
        results = processor.process_zarr_data(
            zarr_path=sample_temperature_zarr,
            gdf=gdf,
            scenario='historical',
            variable='tas',
            threshold=0.0,  # Not used for temperature
            chunk_by_county=True
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'mean_annual_temp_c' in results.columns
        assert set(results['scenario'].unique()) == {'historical'}
    
    def test_county_processor_context_manager(self, sample_counties_shapefile, sample_precipitation_zarr):
        """Test ModernCountyProcessor as context manager."""
        with ModernCountyProcessor(n_workers=2) as processor:
            # Load shapefile
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            # Process precipitation data
            results = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                scenario='test',
                variable='pr',
                threshold=25.4
            )
            
            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0


class TestCLIIntegration:
    """Test CLI integration with the modular system."""
    
    def test_cli_module_exists(self):
        """Test CLI module exists."""
        assert hasattr(county_stats_cli, 'main')
        assert callable(county_stats_cli.main)
    
    def test_cli_precipitation_processing(self, sample_counties_shapefile, sample_precipitation_zarr, temp_dir):
        """Test CLI precipitation processing using direct processor call."""
        output_path = Path(temp_dir) / "cli_precip_results.csv"
        
        # Test the processor that the CLI would use
        with ModernCountyProcessor(n_workers=2) as processor:
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            results = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                variable='pr',
                scenario='ssp370',
                threshold=25.4,
                chunk_by_county=False
            )
            
            # Save results like CLI would
            results.to_csv(output_path, index=False)
        
        # Check output file
        assert output_path.exists()
        
        # Load and validate results
        results_loaded = pd.read_csv(output_path)
        assert len(results_loaded) > 0
        assert 'total_annual_precip_mm' in results_loaded.columns
        assert set(results_loaded['scenario'].unique()) == {'ssp370'}
    
    def test_cli_temperature_processing(self, sample_counties_shapefile, sample_temperature_zarr, temp_dir):
        """Test CLI temperature processing using direct processor call."""
        output_path = Path(temp_dir) / "cli_temp_results.csv"
        
        # Test the processor that the CLI would use
        with ModernCountyProcessor(n_workers=2) as processor:
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            results = processor.process_zarr_data(
                zarr_path=sample_temperature_zarr,
                gdf=gdf,
                variable='tas',
                scenario='historical',
                threshold=0.0,  # Not used for temperature
                chunk_by_county=True
            )
            
            # Save results like CLI would
            results.to_csv(output_path, index=False)
        
        # Check output file
        assert output_path.exists()
        
        # Load and validate results
        results_loaded = pd.read_csv(output_path)
        assert len(results_loaded) > 0
        assert 'mean_annual_temp_c' in results_loaded.columns
        assert set(results_loaded['scenario'].unique()) == {'historical'}


class TestBackwardCompatibilityIntegration:
    """Test that the modular system maintains backward compatibility."""
    
    def test_import_compatibility(self):
        """Test that all expected imports work."""
        # Test main processor import
        from climate_zarr import ModernCountyProcessor
        assert ModernCountyProcessor is not None
        
        # Test new modular imports
        from climate_zarr.county_processor import ModernCountyProcessor
        from climate_zarr import county_stats_cli
        assert ModernCountyProcessor is not None
        assert county_stats_cli is not None
        
        # Test processor imports
        from climate_zarr.processors import (
            PrecipitationProcessor,
            TemperatureProcessor,
            TasMaxProcessor,
            TasMinProcessor
        )
        assert PrecipitationProcessor is not None
        assert TemperatureProcessor is not None
        assert TasMaxProcessor is not None
        assert TasMinProcessor is not None
    
    def test_api_compatibility(self, sample_counties_shapefile, sample_precipitation_zarr):
        """Test that the API remains compatible."""
        # Test that old-style usage still works
        with ModernCountyProcessor(n_workers=2) as processor:
            # These methods should still exist and work
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            assert isinstance(gdf, gpd.GeoDataFrame)
            
            results = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                scenario='test',
                variable='pr',
                threshold=25.4
            )
            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0
    
    def test_method_signatures_compatibility(self):
        """Test that method signatures remain compatible."""
        processor = ModernCountyProcessor()
        
        # Test that expected methods exist
        assert hasattr(processor, 'prepare_shapefile')
        assert hasattr(processor, 'process_zarr_data')
        assert hasattr(processor, 'close')
        assert hasattr(processor, '__enter__')
        assert hasattr(processor, '__exit__')
        
        # Test that methods have expected signatures
        import inspect
        
        # Check prepare_shapefile signature
        sig = inspect.signature(processor.prepare_shapefile)
        assert 'shapefile_path' in sig.parameters
        
        # Check process_zarr_data signature
        sig = inspect.signature(processor.process_zarr_data)
        expected_params = ['zarr_path', 'gdf', 'scenario', 'variable']
        for param in expected_params:
            assert param in sig.parameters


class TestPerformanceIntegration:
    """Test performance aspects of the modular system."""
    
    def test_memory_usage_integration(self, sample_counties_shapefile, sample_precipitation_zarr):
        """Test that memory usage is reasonable."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with ModernCountyProcessor(n_workers=2) as processor:
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            # Process data
            results = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                scenario='test',
                variable='pr',
                threshold=25.4,
                chunk_by_county=False  # Use ultra-fast strategy
            )
            
            # Check memory usage during processing
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Memory increase should be reasonable (less than 500MB for test data)
            memory_increase = peak_memory - initial_memory
            assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"
            
            # Results should be valid
            assert len(results) > 0
    
    def test_processing_speed_integration(self, sample_counties_shapefile, sample_precipitation_zarr):
        """Test that processing completes in reasonable time."""
        import time
        
        start_time = time.time()
        
        with ModernCountyProcessor(n_workers=2) as processor:
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            results = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                scenario='test',
                variable='pr',
                threshold=25.4,
                chunk_by_county=False
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (30 seconds for test data)
        assert processing_time < 30, f"Processing took {processing_time:.1f} seconds"
        
        # Should produce valid results
        assert len(results) > 0
    
    def test_strategy_performance_comparison(self, sample_counties_shapefile, sample_precipitation_zarr):
        """Test that different strategies have expected performance characteristics."""
        import time
        
        with ModernCountyProcessor(n_workers=2) as processor:
            gdf = processor.prepare_shapefile(sample_counties_shapefile)
            
            # Test vectorized strategy
            start_time = time.time()
            results_vectorized = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                scenario='test_vectorized',
                variable='pr',
                threshold=25.4,
                chunk_by_county=True
            )
            vectorized_time = time.time() - start_time
            
            # Test ultra-fast strategy
            start_time = time.time()
            results_ultrafast = processor.process_zarr_data(
                zarr_path=sample_precipitation_zarr,
                gdf=gdf,
                scenario='test_ultrafast',
                variable='pr',
                threshold=25.4,
                chunk_by_county=False
            )
            ultrafast_time = time.time() - start_time
            
            # Both should complete
            assert len(results_vectorized) > 0
            assert len(results_ultrafast) > 0
            
            # Both should complete within reasonable time
            assert vectorized_time < 30
            assert ultrafast_time < 30
            
            # For this small dataset, times should be comparable
            # (Ultra-fast advantage shows up with larger datasets)
            assert vectorized_time > 0
            assert ultrafast_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 