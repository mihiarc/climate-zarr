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
        # County identifier
        if 'GEOID' in gdf.columns:
            gdf['county_id'] = gdf['GEOID']
        elif 'FIPS' in gdf.columns:
            gdf['county_id'] = gdf['FIPS']
        else:
            gdf['county_id'] = gdf.index.astype(str)
        
        # County name
        if 'NAME' in gdf.columns:
            gdf['county_name'] = gdf['NAME']
        elif 'NAMELSAD' in gdf.columns:
            gdf['county_name'] = gdf['NAMELSAD']
        else:
            gdf['county_name'] = gdf['county_id']
        
        # State - handle STATEFP (FIPS codes) and convert to abbreviations
        if 'STUSPS' in gdf.columns:
            gdf['state'] = gdf['STUSPS']
        elif 'STATE_NAME' in gdf.columns:
            gdf['state'] = gdf['STATE_NAME']
        elif 'STATEFP' in gdf.columns:
            # Convert FIPS codes to state abbreviations
            state_fips_to_abbr = {
                '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
                '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
                '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
                '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
                '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
                '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
                '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
                '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
                '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
                '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
                '56': 'WY', '60': 'AS', '66': 'GU', '69': 'MP', '72': 'PR',
                '78': 'VI'
            }
            gdf['state'] = gdf['STATEFP'].map(state_fips_to_abbr).fillna('')
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
            DataArray with standardized coordinates
        """
        # Rename dimensions for rioxarray compatibility
        if 'lon' in data.dims and 'lat' in data.dims:
            data = data.rename({'lon': 'x', 'lat': 'y'})
        
        # Add spatial reference
        data = data.rio.write_crs('EPSG:4326')
        
        # Handle longitude wrapping if needed
        if 'x' in data.coords and float(data.x.max()) > 180:
            data = data.assign_coords(x=(data.x + 180) % 360 - 180)
            data = data.sortby('x')
            data = data.rio.write_crs('EPSG:4326')
        
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