#!/usr/bin/env python
"""Base processor class for county-level climate statistics."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union
import warnings

import geopandas as gpd
import xarray as xr
from rich.console import Console

warnings.filterwarnings('ignore', category=RuntimeWarning)

console = Console()


class BaseCountyProcessor(ABC):
    """Base class for county-level climate data processing."""
    
    def __init__(
        self,
        n_workers: int = 4
    ):
        """Initialize the processor.
        
        Args:
            n_workers: Number of worker processes
        """
        self.n_workers = n_workers
    

    
    def prepare_shapefile(
        self, 
        shapefile_path: Path, 
        target_crs: str = 'EPSG:4326'
    ) -> gpd.GeoDataFrame:
        """Load and prepare shapefile for processing.
        
        Args:
            shapefile_path: Path to the shapefile
            target_crs: Target coordinate reference system
            
        Returns:
            Prepared GeoDataFrame with standardized columns
        """
        console.print(f"[blue]Loading shapefile:[/blue] {shapefile_path}")
        
        # Load with optimizations
        gdf = gpd.read_file(shapefile_path)
        
        # Optimize geometry column
        gdf.geometry = gdf.geometry.simplify(0.001)  # Slight simplification for speed
        
        # Convert to target CRS if needed
        if gdf.crs.to_string() != target_crs:
            console.print(f"[yellow]Converting CRS from {gdf.crs} to {target_crs}[/yellow]")
            gdf = gdf.to_crs(target_crs)
        
        # Standardize column names
        gdf = self._standardize_columns(gdf)
        
        # Add spatial index for faster operations
        gdf.sindex  # Force creation of spatial index
        
        return gdf
    
    def _standardize_columns(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardize column names and add required fields.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            GeoDataFrame with standardized columns
        """
        # County identifier - ensure FIPS codes are formatted as 5-character strings with zero padding
        if 'GEOID' in gdf.columns:
            # Convert to string and pad with zeros to ensure 5 characters for mapping compatibility
            gdf['county_id'] = gdf['GEOID'].astype(str).str.zfill(5)
        elif 'FIPS' in gdf.columns:
            # Convert to string and pad with zeros to ensure 5 characters for mapping compatibility
            gdf['county_id'] = gdf['FIPS'].astype(str).str.zfill(5)
        else:
            # Use index as fallback, zero-padded to 5 characters
            gdf['county_id'] = gdf.index.astype(str).str.zfill(5)
        
        # County name
        if 'NAME' in gdf.columns:
            gdf['county_name'] = gdf['NAME']
        elif 'NAMELSAD' in gdf.columns:
            gdf['county_name'] = gdf['NAMELSAD']
        else:
            gdf['county_name'] = gdf['county_id']
        
        # State
        if 'STUSPS' in gdf.columns:
            gdf['state'] = gdf['STUSPS']
        elif 'STATE_NAME' in gdf.columns:
            gdf['state'] = gdf['STATE_NAME']
        else:
            gdf['state'] = ''
        
        # Add numeric index for vectorized operations
        gdf['raster_id'] = range(1, len(gdf) + 1)
        
        return gdf[['county_id', 'county_name', 'state', 'raster_id', 'geometry']]
    
    def _standardize_coordinates(self, data: xr.DataArray) -> xr.DataArray:
        """Standardize coordinate system and spatial reference.
        
        Args:
            data: Input xarray DataArray
            
        Returns:
            DataArray with standardized coordinates and proper labeling
        """
        # Rename dimensions for rioxarray compatibility
        if 'lon' in data.dims and 'lat' in data.dims:
            data = data.rename({'lon': 'x', 'lat': 'y'})
        
        # Ensure spatial reference is set
        try:
            if not hasattr(data, 'rio') or data.rio.crs is None:
                data = data.rio.write_crs('EPSG:4326')
        except Exception:
            console.print("[yellow]Warning: Could not set spatial reference[/yellow]")
        
        # Handle longitude wrapping if needed
        if 'x' in data.coords and float(data.x.max()) > 180:
            data = data.assign_coords(x=(data.x + 180) % 360 - 180)
            data = data.sortby('x')
            try:
                data = data.rio.write_crs('EPSG:4326')
            except Exception:
                console.print("[yellow]Warning: Could not re-set spatial reference after longitude wrapping[/yellow]")
        
        # Ensure coordinates remain labeled after rioxarray operations
        data = self._ensure_coordinate_labels(data)
        
        return data
    
    def _ensure_coordinate_labels(self, data: xr.DataArray) -> xr.DataArray:
        """Ensure coordinate labels are preserved after rioxarray operations.
        
        Args:
            data: xarray DataArray that may have unlabeled dimensions
            
        Returns:
            DataArray with properly labeled coordinates
        """
        # Check if we have unlabeled dimensions (common after rio operations)
        unlabeled_dims = []
        for dim in data.dims:
            if dim not in data.coords:
                unlabeled_dims.append(dim)
        
        if unlabeled_dims:
            console.print(f"[yellow]Fixing unlabeled dimensions: {unlabeled_dims}[/yellow]")
            
            # Recreate coordinate arrays for unlabeled dimensions
            new_coords = {}
            for dim in unlabeled_dims:
                dim_size = data.sizes[dim]
                if dim == 'y' or dim == 'lat':
                    # Create latitude-like coordinates (decreasing from north to south)
                    new_coords[dim] = np.linspace(90, -90, dim_size)
                elif dim == 'x' or dim == 'lon':
                    # Create longitude-like coordinates (increasing from west to east)
                    new_coords[dim] = np.linspace(-180, 180, dim_size)
                elif dim == 'time':
                    # Create generic time coordinates
                    new_coords[dim] = np.arange(dim_size)
                else:
                    # Generic numeric coordinates for other dimensions
                    new_coords[dim] = np.arange(dim_size)
            
            # Assign the new coordinates
            if new_coords:
                data = data.assign_coords(new_coords)
        
        return data
    
    @abstractmethod
    def process_variable_data(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        **kwargs
    ) -> Dict:
        """Process climate variable data for all counties.
        
        This method must be implemented by subclasses for specific variables.
        
        Args:
            data: Climate data array
            gdf: County geometries
            scenario: Scenario name
            **kwargs: Additional variable-specific parameters
            
        Returns:
            Dictionary containing processed results
        """
        pass
    
    def close(self):
        """Clean up resources."""
        pass
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
        return False 