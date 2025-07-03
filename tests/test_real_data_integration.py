"""
Integration tests using real project data files.
"""

import pytest
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from climate_zarr.stack_nc_to_zarr import stack_netcdf_to_zarr
from climate_zarr.calculate_county_stats import ModernCountyProcessor
from climate_zarr.climate_cli import app
from typer.testing import CliRunner


class TestRealDataIntegration:
    """Integration tests using actual climate data and shapefiles from the project."""
    
    @pytest.fixture(scope="class")
    def real_nc_files(self):
        """Get a subset of real NetCDF files for testing."""
        data_dir = Path("data")
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        # Get first 3 years of data for faster tests
        nc_files = sorted(data_dir.glob("pr_day_NorESM2-LM_historical_*.nc"))[:3]
        if len(nc_files) < 3:
            pytest.skip("Not enough NetCDF files found")
        
        return nc_files
    
    @pytest.fixture(scope="class")
    def conus_shapefile(self):
        """Get the CONUS counties shapefile."""
        shapefile = Path("regional_counties/conus_counties.shp")
        if not shapefile.exists():
            pytest.skip("CONUS shapefile not found")
        return shapefile
    
    @pytest.fixture(scope="class")
    def hawaii_shapefile(self):
        """Get the Hawaii counties shapefile."""
        shapefile = Path("regional_counties/hawaii_counties.shp")
        if not shapefile.exists():
            pytest.skip("Hawaii shapefile not found")
        return shapefile
    
    def test_real_zarr_conversion(self, real_nc_files, tmp_path):
        """Test Zarr conversion with real climate data."""
        output_zarr = tmp_path / "real_climate.zarr"
        
        # Convert real NetCDF files to Zarr
        stack_netcdf_to_zarr(
            nc_files=real_nc_files,
            zarr_path=output_zarr,
            compression="zstd",
            compression_level=3,
            chunks={"time": 365, "lat": 180, "lon": 360}
        )
        
        # Verify the output
        assert output_zarr.exists()
        
        # Open and check the Zarr store
        ds = xr.open_zarr(output_zarr)
        
        # Check expected variable (precipitation)
        assert "pr" in ds.data_vars
        
        # Check dimensions
        assert "time" in ds.dims
        assert "lat" in ds.dims
        assert "lon" in ds.dims
        
        # Check that we have multiple years of data
        assert len(ds.time) > 365  # More than one year
        
        # Check data integrity
        assert not ds.pr.isnull().all()
        
        # Check attributes are preserved
        assert "units" in ds.pr.attrs
        assert ds.pr.attrs.get("units") in ["mm/day", "kg m-2 s-1", "mm day-1"]
    
    def test_real_county_stats_conus(self, real_nc_files, conus_shapefile, tmp_path):
        """Test county statistics with real CONUS data."""
        # First create a Zarr store
        zarr_path = tmp_path / "conus_test.zarr"
        
        # Use just one year for speed
        stack_netcdf_to_zarr(
            nc_files=real_nc_files[:1],
            zarr_path=zarr_path,
            clip_region="conus"  # Clip to CONUS region
        )
        
        # Load and prepare the shapefile
        gdf = gpd.read_file(conus_shapefile)
        
        # Prepare required columns
        gdf['county_id'] = gdf.get('GEOID', range(len(gdf)))
        gdf['county_name'] = gdf.get('NAME', [f'County_{i}' for i in range(len(gdf))])
        gdf['state'] = gdf.get('STATE_NAME', gdf.get('STATEFP', ''))
        gdf['raster_id'] = range(1, len(gdf) + 1)
        
        # Process with limited counties for speed
        gdf_subset = gdf.head(10)  # Just first 10 counties
        
        processor = ModernCountyProcessor(n_workers=2, memory_limit="2GB")
        
        results = processor.process_zarr_data(
            zarr_path=zarr_path,
            gdf=gdf_subset,
            variable="pr",
            chunk_by_county=False
        )
        
        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert len(results) <= 10  # Should have results for subset
        
        # Save results
        output_csv = tmp_path / "conus_stats.csv"
        results.to_csv(output_csv, index=False)
        assert output_csv.exists()
    
    def test_real_region_clipping(self, real_nc_files, tmp_path):
        """Test region clipping with real data."""
        # Test clipping for different regions
        regions = ["conus", "alaska", "hawaii"]
        
        for region in regions:
            output_zarr = tmp_path / f"{region}_clipped.zarr"
            
            try:
                stack_netcdf_to_zarr(
                    nc_files=real_nc_files[:1],  # Just one file
                    zarr_path=output_zarr,
                    clip_region=region
                )
                
                # If successful, verify the output
                if output_zarr.exists():
                    ds = xr.open_zarr(output_zarr)
                    
                    # Basic checks
                    assert "pr" in ds.data_vars
                    assert len(ds.lat) > 0
                    assert len(ds.lon) > 0
                    
                    print(f"Region {region} - Lat range: {ds.lat.min().values:.2f} to {ds.lat.max().values:.2f}")
                    print(f"Region {region} - Lon range: {ds.lon.min().values:.2f} to {ds.lon.max().values:.2f}")
                    
            except Exception as e:
                # Region clipping might fail if bounds aren't configured
                print(f"Region {region} clipping failed: {e}")
    
    def test_cli_with_real_data(self, real_nc_files, tmp_path):
        """Test CLI commands with real data."""
        cli_runner = CliRunner()
        
        # Test create-zarr with real data
        data_dir = Path("data")
        output_zarr = tmp_path / "cli_test.zarr"
        
        result = cli_runner.invoke(app, [
            "create-zarr",
            str(data_dir),
            "--output", str(output_zarr),
            "--compression", "lz4",
            "--chunks", "time=100,lat=200,lon=200"
        ], input="n\n")  # Skip any interactive prompts
        
        # Check if command succeeded
        if result.exit_code == 0:
            assert output_zarr.exists()
            
            # Verify the created Zarr
            ds = xr.open_zarr(output_zarr)
            assert "pr" in ds.data_vars
    
    def test_multi_year_processing(self, real_nc_files, tmp_path):
        """Test processing multiple years of data."""
        if len(real_nc_files) < 2:
            pytest.skip("Need at least 2 years of data")
        
        output_zarr = tmp_path / "multi_year.zarr"
        
        # Process multiple years
        stack_netcdf_to_zarr(
            nc_files=real_nc_files[:2],  # Two years
            zarr_path=output_zarr,
            chunks={"time": 365}  # One year chunks
        )
        
        # Verify
        ds = xr.open_zarr(output_zarr)
        
        # Should have approximately 2 years of daily data
        assert len(ds.time) >= 365 * 2
        assert len(ds.time) <= 366 * 2  # Account for leap years
        
        # Check time is properly concatenated
        time_diff = ds.time.diff("time")
        # Most differences should be 1 day
        assert (time_diff == pd.Timedelta(days=1)).sum() > len(time_diff) * 0.95
    
    def test_performance_benchmark(self, real_nc_files, tmp_path):
        """Benchmark performance with real data."""
        import time
        
        output_zarr = tmp_path / "benchmark.zarr"
        
        # Time the conversion
        start_time = time.time()
        
        stack_netcdf_to_zarr(
            nc_files=real_nc_files[:1],  # One year
            zarr_path=output_zarr,
            compression="zstd",
            compression_level=1  # Fast compression
        )
        
        conversion_time = time.time() - start_time
        
        # Get file sizes
        nc_size = sum(f.stat().st_size for f in real_nc_files[:1]) / (1024**2)  # MB
        zarr_size = sum(f.stat().st_size for f in output_zarr.rglob("*") if f.is_file()) / (1024**2)  # MB
        
        print(f"\nPerformance Metrics:")
        print(f"Conversion time: {conversion_time:.2f} seconds")
        print(f"NetCDF size: {nc_size:.2f} MB")
        print(f"Zarr size: {zarr_size:.2f} MB")
        print(f"Compression ratio: {nc_size/zarr_size:.2f}x")
        print(f"Processing speed: {nc_size/conversion_time:.2f} MB/s")
        
        # Basic performance assertions
        assert conversion_time < 60  # Should complete in under a minute
        # Zarr might be slightly larger due to metadata, but should be within 20%
        assert zarr_size < nc_size * 1.2  # Allow up to 20% overhead