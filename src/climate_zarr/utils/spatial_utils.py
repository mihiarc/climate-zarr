#!/usr/bin/env python
"""Spatial processing utilities for climate data."""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rich.console import Console

console = Console()


def create_county_raster(
    gdf: gpd.GeoDataFrame,
    lats: np.ndarray,
    lons: np.ndarray
) -> np.ndarray:
    """Create a raster mask for counties.
    
    Args:
        gdf: GeoDataFrame with county geometries
        lats: Latitude coordinate array
        lons: Longitude coordinate array
        
    Returns:
        County raster mask array
    """
    console.print("[cyan]Creating county raster mask...[/cyan]")
    
    # Create transform for the zarr grid
    transform = from_bounds(
        lons.min(), lats.min(), lons.max(), lats.max(),
        len(lons), len(lats)
    )
    
    # Add unique IDs to counties for rasterization
    gdf_with_ids = gdf.copy()
    if 'raster_id' not in gdf_with_ids.columns:
        gdf_with_ids['raster_id'] = range(1, len(gdf) + 1)
    
    # Create shapes for rasterization
    shapes = [(geom, raster_id) for geom, raster_id in 
             zip(gdf_with_ids.geometry, gdf_with_ids.raster_id)]
    
    # Rasterize counties to create mask
    county_raster = rasterize(
        shapes,
        out_shape=(len(lats), len(lons)),
        transform=transform,
        fill=0,
        dtype='uint16'
    )
    
    unique_counties = np.unique(county_raster[county_raster > 0])
    console.print(f"[cyan]County raster created with {len(unique_counties)} counties[/cyan]")
    
    return county_raster


def get_time_information(data: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Extract time information from xarray DataArray.
    
    Args:
        data: xarray DataArray with time dimension
        
    Returns:
        Tuple of (years array, unique years array)
    """
    time_values = data.time.values
    
    if hasattr(time_values[0], 'year'):
        years = np.array([t.year for t in time_values])
    else:
        years = pd.to_datetime(time_values).year.values
    
    unique_years = np.unique(years)
    
    return years, unique_years


def get_coordinate_arrays(data: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Extract coordinate arrays from xarray DataArray.
    
    Args:
        data: xarray DataArray with spatial coordinates
        
    Returns:
        Tuple of (lats, lons) coordinate arrays
    """
    # Handle different coordinate names
    if 'lat' in data.coords:
        lats = data.lat.values
        lons = data.lon.values
    elif 'y' in data.coords:
        lats = data.y.values
        lons = data.x.values
    else:
        coord_names = list(data.coords)
        raise ValueError(f"Could not find lat/lon coordinates in {coord_names}")
    
    return lats, lons


def clip_county_data(
    data: xr.DataArray,
    county_geometry,
    all_touched: bool = True
) -> xr.DataArray:
    """Clip data to a county geometry using rioxarray.
    
    Args:
        data: xarray DataArray with spatial coordinates
        county_geometry: Shapely geometry for the county
        all_touched: Whether to include all touched pixels
        
    Returns:
        Clipped DataArray
    """
    return data.rio.clip([county_geometry], all_touched=all_touched) 