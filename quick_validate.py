#!/usr/bin/env python3
"""Quick validation to extract raw climate data for quality checking."""

import sys
from pathlib import Path
import xarray as xr
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.climate_utils import adjust_longitude_bounds
from src.utils.file_operations import load_netcdf

def quick_extract():
    """Extract one year of data for LA County to validate values."""
    
    base_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    
    # LA County bounds
    bounds = [-118.9517, 33.7037, -117.6462, 34.8233]
    min_lon, min_lat, max_lon, max_lat = adjust_longitude_bounds(bounds)
    
    print(f"LA County bounds (adjusted): {min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f}")
    
    # Load one file for each variable for year 2005
    variables = {
        'tasmax': 'Maximum Temperature',
        'tasmin': 'Minimum Temperature', 
        'tas': 'Mean Temperature',
        'pr': 'Precipitation'
    }
    
    for var, desc in variables.items():
        print(f"\n{desc} ({var}):")
        
        # Find 2005 file (try multiple patterns)
        patterns = [
            f"{var}_*_2005.nc",
            f"{var}_*_2005_*.nc"
        ]
        files = []
        for pattern in patterns:
            files = list((base_path / var / "historical").glob(pattern))
            if files:
                break
        
        if files:
            file_path = files[0]
            print(f"  File: {file_path.name}")
            
            # Load with spatial selection
            ds = load_netcdf(
                file_path,
                preselect_bounds={
                    'lat': slice(min_lat - 0.1, max_lat + 0.1),
                    'lon': slice(min_lon - 0.1, max_lon + 0.1)
                }
            )
            
            if ds and var in ds:
                data = ds[var]
                print(f"  Shape: {data.shape}")
                print(f"  Units: {data.attrs.get('units', 'unknown')}")
                
                # Calculate spatial mean for a few days
                weights = np.cos(np.deg2rad(data.lat))
                spatial_mean = data.weighted(weights).mean(dim=['lat', 'lon'])
                
                # Show first 5 days
                print(f"  First 5 days (spatial average):")
                for i in range(min(5, len(spatial_mean))):
                    value = float(spatial_mean.isel(time=i).values)
                    date = str(spatial_mean.time.isel(time=i).values)[:10]
                    
                    if var in ['tas', 'tasmax', 'tasmin']:
                        # Convert from K to C and F
                        celsius = value - 273.15
                        fahrenheit = celsius * 9/5 + 32
                        print(f"    {date}: {value:.2f} K = {celsius:.2f}°C = {fahrenheit:.2f}°F")
                    else:
                        # Convert precipitation from kg/m2/s to mm/day
                        mm_day = value * 86400
                        inches_day = mm_day / 25.4
                        print(f"    {date}: {value:.6f} kg/m²/s = {mm_day:.2f} mm/day = {inches_day:.2f} in/day")
                
                # Annual statistics
                if var in ['tas', 'tasmax', 'tasmin']:
                    annual_mean_k = float(spatial_mean.mean().values)
                    annual_mean_c = annual_mean_k - 273.15
                    annual_mean_f = annual_mean_c * 9/5 + 32
                    print(f"  Annual mean: {annual_mean_c:.2f}°C ({annual_mean_f:.2f}°F)")
                    
                    if var == 'tasmax':
                        # Count days above 90F (305.37 K)
                        hot_days = (spatial_mean > 305.37).sum().values
                        print(f"  Days above 90°F: {hot_days}")
                    elif var == 'tasmin':
                        # Count nights below 32F (273.15 K)
                        frost_nights = (spatial_mean < 273.15).sum().values
                        print(f"  Nights below 32°F: {frost_nights}")
                        
                else:  # precipitation
                    # Total annual precipitation
                    annual_total_kg = float(spatial_mean.sum().values) * 86400  # kg/m²
                    annual_total_mm = annual_total_kg  # kg/m² = mm
                    annual_total_inches = annual_total_mm / 25.4
                    print(f"  Annual total: {annual_total_mm:.1f} mm ({annual_total_inches:.1f} inches)")
                    
                    # Days with precipitation > 25.4 mm (1 inch)
                    heavy_rain_days = (spatial_mean * 86400 > 25.4).sum().values
                    print(f"  Days with >1 inch rain: {heavy_rain_days}")
            else:
                print(f"  ERROR: Could not load data")
        else:
            print(f"  ERROR: No files found for {var} in 2005")

if __name__ == "__main__":
    quick_extract()