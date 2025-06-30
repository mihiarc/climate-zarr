#!/usr/bin/env python
"""Test just the data loading functionality."""

import xarray as xr
from pathlib import Path

def test_load_single_file():
    """Test loading a single climate data file."""
    # Pick a single file
    data_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    tas_file = data_path / "tas/historical/tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2010.nc"
    
    print(f"Testing file: {tas_file}")
    print(f"File exists: {tas_file.exists()}")
    
    if tas_file.exists():
        # Load the file
        ds = xr.open_dataset(tas_file)
        
        print(f"\nDataset info:")
        print(f"- Variables: {list(ds.data_vars)}")
        print(f"- Dimensions: {dict(ds.dims)}")
        print(f"- Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        print(f"- Time calendar: {ds.time.encoding.get('calendar', 'standard')}")
        
        # Check coordinate ranges
        print(f"\nCoordinate ranges:")
        print(f"- Latitude: {ds.lat.min().values:.2f} to {ds.lat.max().values:.2f}")
        print(f"- Longitude: {ds.lon.min().values:.2f} to {ds.lon.max().values:.2f}")
        
        # Test selecting a small region (Nebraska approx bounds)
        subset = ds.sel(lat=slice(40, 43), lon=slice(263, 267))
        print(f"\nSubset for Nebraska region:")
        print(f"- Shape: {subset.tas.shape}")
        print(f"- Size: {subset.tas.size} values")
        
        ds.close()
        
        return True
    
    return False

if __name__ == "__main__":
    success = test_load_single_file()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")