#!/usr/bin/env python3
"""
Debug version to understand what's happening in parallel processing
"""

import xarray as xr
import numpy as np
from pathlib import Path

def debug_data_extraction():
    """
    Test the data extraction approach used in parallel processor
    """
    print("DEBUG: Testing data extraction method")
    print("="*60)
    
    # Test with one county
    base_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    
    # Load a sample file
    print("\n1. Loading sample data file...")
    sample_file = base_path / "tas" / "historical" / "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2009.nc"
    ds = xr.open_dataset(sample_file)
    
    print(f"   Data shape: {ds['tas'].shape}")
    print(f"   Lat range: {ds.lat.min().values:.2f} to {ds.lat.max().values:.2f}")
    print(f"   Lon range: {ds.lon.min().values:.2f} to {ds.lon.max().values:.2f}")
    
    # Test county bounds approach (Los Angeles)
    # LA County approximate bounds: lat 33.7-34.8, lon -118.9 to -117.6
    print("\n2. Testing county bounds extraction (Los Angeles)...")
    county_bounds = (33.7, -118.9, 34.8, -117.6)  # min_lat, min_lon, max_lat, max_lon
    
    # The parallel processor uses this approach:
    lat_slice = slice(county_bounds[0] - 0.5, county_bounds[2] + 0.5)
    lon_slice = slice(county_bounds[1] - 0.5, county_bounds[3] + 0.5)
    
    print(f"   Lat slice: {lat_slice}")
    print(f"   Lon slice: {lon_slice}")
    
    # Try to extract
    try:
        county_data = ds['tas'].sel(lat=lat_slice, lon=lon_slice)
        print(f"   Extracted shape: {county_data.shape}")
        print(f"   Has data: {not np.all(np.isnan(county_data.values))}")
        
        # Check if we got any data
        if county_data.size == 0:
            print("   WARNING: No data extracted!")
        else:
            # Calculate mean
            weights = np.cos(np.deg2rad(county_data.lat))
            county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
            print(f"   Mean value: {county_mean.values[0]:.2f} K")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test with corrected longitude (0-360 instead of -180 to 180)
    print("\n3. Testing with longitude correction...")
    # Convert negative longitudes to 0-360 range
    lon_min = county_bounds[1] % 360  # -118.9 -> 241.1
    lon_max = county_bounds[3] % 360  # -117.6 -> 242.4
    
    print(f"   Converted lon range: {lon_min:.1f} to {lon_max:.1f}")
    
    lon_slice_corrected = slice(lon_min - 0.5, lon_max + 0.5)
    
    try:
        county_data = ds['tas'].sel(lat=lat_slice, lon=lon_slice_corrected)
        print(f"   Extracted shape: {county_data.shape}")
        print(f"   Has data: {not np.all(np.isnan(county_data.values))}")
        
        if county_data.size > 0:
            weights = np.cos(np.deg2rad(county_data.lat))
            county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
            print(f"   Mean value: {county_mean.values[0]:.2f} K ({county_mean.values[0] - 273.15:.2f}°C)")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test alternative: use nearest neighbor selection
    print("\n4. Testing nearest neighbor approach...")
    center_lat = (county_bounds[0] + county_bounds[2]) / 2
    center_lon = (county_bounds[1] + county_bounds[3]) / 2
    center_lon_360 = center_lon % 360
    
    print(f"   County center: {center_lat:.2f}°N, {center_lon:.2f}°E")
    print(f"   Center lon (0-360): {center_lon_360:.2f}")
    
    try:
        # Get nearest grid points
        nearest_data = ds['tas'].sel(lat=center_lat, lon=center_lon_360, method='nearest')
        print(f"   Nearest point value: {nearest_data.values[0]:.2f} K ({nearest_data.values[0] - 273.15:.2f}°C)")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    ds.close()
    
    # Now test with actual county shapefile bounds
    print("\n5. Loading actual county bounds from shapefile...")
    import geopandas as gpd
    
    counties = gpd.read_file("/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp")
    counties = counties.to_crs('EPSG:4326')
    
    # Get LA County
    la_county = counties[counties['GEOID'] == '06037'].iloc[0]
    actual_bounds = la_county.geometry.bounds
    print(f"   Actual LA County bounds: {actual_bounds}")
    print(f"   Geometry type: {la_county.geometry.geom_type}")
    
    # Test with regionmask approach (as in simple test)
    print("\n6. Testing regionmask approach...")
    import regionmask
    
    test_counties = counties[counties['GEOID'].isin(['06037'])].copy()
    counties_mask = regionmask.from_geopandas(
        test_counties,
        names="GEOID",
        abbrevs="GEOID",
        name="Test_County"
    )
    
    # Reload data
    ds = xr.open_dataset(sample_file)
    mask_3D = counties_mask.mask_3D(ds)
    county_mask = mask_3D.isel(region=0)
    county_data = ds['tas'].where(county_mask)
    
    # Check if we have data
    valid_points = (~np.isnan(county_data.values)).sum()
    print(f"   Valid grid points in county: {valid_points}")
    
    if valid_points > 0:
        weights = np.cos(np.deg2rad(ds.lat))
        county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
        value = county_mean.compute().values[0]
        print(f"   Mean value: {value:.2f} K ({value - 273.15:.2f}°C)")
    
    ds.close()

if __name__ == "__main__":
    debug_data_extraction()