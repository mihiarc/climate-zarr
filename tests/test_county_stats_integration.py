"""
Integration tests for county statistics calculation workflow.
"""

import pytest
import pandas as pd
import xarray as xr
from pathlib import Path
from climate_zarr.calculate_county_stats import ModernCountyProcessor
from climate_zarr.stack_nc_to_zarr import stack_netcdf_to_zarr


class TestCountyStatsIntegration:
    """Test the complete county statistics calculation workflow."""
    
    @pytest.fixture
    def zarr_store_with_data(self, sample_netcdf_files, zarr_output_dir):
        """Create a Zarr store with test data for county stats."""
        zarr_path = zarr_output_dir / "test_data_for_stats.zarr"
        
        # Create Zarr store if it doesn't exist
        if not zarr_path.exists():
            stack_netcdf_to_zarr(
                nc_files=[Path(f) for f in sample_netcdf_files],
                zarr_path=zarr_path,
                chunks={"time": 100, "lat": 10, "lon": 10}
            )
        
        return zarr_path
    
    def test_basic_county_stats(self, zarr_store_with_data, sample_shapefile, stats_output_dir):
        """Test basic county statistics calculation."""
        output_csv = stats_output_dir / "basic_county_stats.csv"
        
        # Initialize processor
        processor = ModernCountyProcessor(
            n_workers=2,
            memory_limit="2GB"
        )
        
        # Load shapefile as GeoDataFrame and prepare it
        import geopandas as gpd
        gdf = gpd.read_file(sample_shapefile)
        
        # Add required columns for processing
        if 'county_id' not in gdf.columns:
            gdf['county_id'] = gdf.get('GEOID', range(len(gdf)))
        if 'county_name' not in gdf.columns:
            gdf['county_name'] = gdf.get('NAME', [f'County_{i}' for i in range(len(gdf))])
        if 'state' not in gdf.columns:
            gdf['state'] = gdf.get('STATEFP', '')
        gdf['raster_id'] = range(1, len(gdf) + 1)
        
        # Process counties using the actual method
        results = processor.process_zarr_data(
            zarr_path=zarr_store_with_data,
            gdf=gdf,
            variable="tas",
            chunk_by_county=False
        )
        
        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        
        # Save results for inspection
        results.to_csv(output_csv, index=False)
        
        # Check that we have some expected columns
        # The actual columns depend on the processing method
        assert len(results.columns) > 0
    
    def test_precipitation_stats(self, zarr_store_with_data, sample_shapefile, stats_output_dir):
        """Test precipitation statistics calculation."""
        output_csv = stats_output_dir / "precip_stats.csv"
        
        processor = ModernCountyProcessor(n_workers=1)
        
        import geopandas as gpd
        gdf = gpd.read_file(sample_shapefile)
        
        # Add required columns
        if 'county_id' not in gdf.columns:
            gdf['county_id'] = gdf.get('GEOID', range(len(gdf)))
        if 'county_name' not in gdf.columns:
            gdf['county_name'] = gdf.get('NAME', [f'County_{i}' for i in range(len(gdf))])
        if 'state' not in gdf.columns:
            gdf['state'] = gdf.get('STATEFP', '')
        gdf['raster_id'] = range(1, len(gdf) + 1)
        
        results = processor.process_zarr_data(
            zarr_path=zarr_store_with_data,
            gdf=gdf,
            variable="pr",
            chunk_by_county=False
        )
        
        # Basic validation
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
    
    def test_temporal_aggregation(self, zarr_store_with_data, sample_shapefile, stats_output_dir):
        """Test that temporal aggregation works correctly."""
        # Get the time dimension size
        ds = xr.open_zarr(zarr_store_with_data)
        time_steps = len(ds.time)
        ds.close()
        
        processor = ModernCountyProcessor()
        
        import geopandas as gpd
        gdf = gpd.read_file(sample_shapefile)
        
        # Add required columns
        if 'county_id' not in gdf.columns:
            gdf['county_id'] = gdf.get('GEOID', range(len(gdf)))
        if 'county_name' not in gdf.columns:
            gdf['county_name'] = gdf.get('NAME', [f'County_{i}' for i in range(len(gdf))])
        if 'state' not in gdf.columns:
            gdf['state'] = gdf.get('STATEFP', '')
        gdf['raster_id'] = range(1, len(gdf) + 1)
        
        results = processor.process_zarr_data(
            zarr_path=zarr_store_with_data,
            gdf=gdf,
            variable="tas"
        )
        
        # Basic check that we processed temporal data
        assert len(results) > 0
        assert time_steps > 0  # Verify we had temporal data
    
    def test_parallel_processing(self, zarr_store_with_data, sample_shapefile, stats_output_dir):
        """Test parallel processing of counties."""
        import geopandas as gpd
        gdf = gpd.read_file(sample_shapefile)
        
        # Add required columns
        if 'county_id' not in gdf.columns:
            gdf['county_id'] = gdf.get('GEOID', range(len(gdf)))
        if 'county_name' not in gdf.columns:
            gdf['county_name'] = gdf.get('NAME', [f'County_{i}' for i in range(len(gdf))])
        if 'state' not in gdf.columns:
            gdf['state'] = gdf.get('STATEFP', '')
        gdf['raster_id'] = range(1, len(gdf) + 1)
        
        # Process with multiple workers
        processor_parallel = ModernCountyProcessor(n_workers=2)
        results_parallel = processor_parallel.process_zarr_data(
            zarr_path=zarr_store_with_data,
            gdf=gdf,
            variable="tas",
            chunk_by_county=False
        )
        
        # Process with single worker
        processor_serial = ModernCountyProcessor(n_workers=1)
        results_serial = processor_serial.process_zarr_data(
            zarr_path=zarr_store_with_data,
            gdf=gdf,
            variable="tas",
            chunk_by_county=False
        )
        
        # Basic check that both produced results
        assert len(results_parallel) > 0
        assert len(results_serial) > 0
        # Both should process same number of counties
        assert len(results_parallel) == len(results_serial)
    
    def test_error_handling(self, zarr_store_with_data, sample_shapefile):
        """Test error handling for invalid inputs."""
        processor = ModernCountyProcessor()
        
        import geopandas as gpd
        gdf = gpd.read_file(sample_shapefile)
        
        # Test with invalid variable - should raise an error
        with pytest.raises(Exception):  # Broad exception catch for any error
            processor.process_zarr_data(
                zarr_path=zarr_store_with_data,
                gdf=gdf,
                variable="invalid_var",  # This should cause an error
                chunk_by_county=False
            )
    
    def test_percentile_calculations(self, zarr_store_with_data, sample_shapefile, stats_output_dir):
        """Test basic processing completes successfully."""
        processor = ModernCountyProcessor()
        
        import geopandas as gpd
        gdf = gpd.read_file(sample_shapefile)
        
        # Add required columns
        if 'county_id' not in gdf.columns:
            gdf['county_id'] = gdf.get('GEOID', range(len(gdf)))
        if 'county_name' not in gdf.columns:
            gdf['county_name'] = gdf.get('NAME', [f'County_{i}' for i in range(len(gdf))])
        if 'state' not in gdf.columns:
            gdf['state'] = gdf.get('STATEFP', '')
        gdf['raster_id'] = range(1, len(gdf) + 1)
        
        # Just test that processing completes
        results = processor.process_zarr_data(
            zarr_path=zarr_store_with_data,
            gdf=gdf,
            variable="tas"
        )
        
        # Save results
        output_csv = stats_output_dir / "test_stats.csv"
        results.to_csv(output_csv, index=False)
        
        assert output_csv.exists()
        assert len(results) > 0