#!/usr/bin/env python
"""Test processing all counties with a small subset of years to verify it works."""

import sys
sys.path.insert(0, '.')

# Import the main processing function
from process_all_counties_parallel import process_year, test_seven_counties_parallel

# Monkey patch to process only 2 years for testing
def test_subset():
    import process_all_counties_parallel as pacp
    
    # Override years for quick test
    original_func = pacp.test_seven_counties_parallel
    
    def test_with_subset():
        # Temporarily override the years
        pacp.historical_years = [2010, 2011]
        pacp.ssp245_years = [2050, 2051]
        
        # Store original values
        data_path = pacp.Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
        
        # Modify the function to use subset
        import geopandas as gpd
        from pathlib import Path
        
        shapefile_path = Path("data/shapefiles/tl_2024_us_county.shp")
        print(f"Loading counties from {shapefile_path}...")
        counties_gdf = gpd.read_file(shapefile_path)
        
        # Use only first 10 counties for testing
        counties_gdf = counties_gdf.head(10)
        
        print(f"Testing with {len(counties_gdf)} counties and 4 years")
        
        # Continue with modified test
        from process_all_counties_parallel import test_seven_counties_parallel
        test_seven_counties_parallel()
    
    test_with_subset()

if __name__ == "__main__":
    # Test with subset first
    print("=== Testing with subset of counties and years ===")
    test_subset()