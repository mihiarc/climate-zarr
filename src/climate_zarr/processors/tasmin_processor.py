#!/usr/bin/env python
"""Daily minimum temperature data processor for county-level statistics."""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import xarray as xr
from rich.console import Console

from .base_processor import BaseCountyProcessor
from .processing_strategies import VectorizedStrategy, UltraFastStrategy
from ..utils.data_utils import convert_units

console = Console()


class TasMinProcessor(BaseCountyProcessor):
    """Processor for daily minimum temperature data (tasmin variable)."""
    
    def process_variable_data(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        chunk_by_county: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Process daily minimum temperature data for all counties.
        
        Args:
            data: Daily minimum temperature data array
            gdf: County geometries
            scenario: Scenario name
            chunk_by_county: Whether to use chunked processing
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with daily minimum temperature statistics
        """
        console.print("[blue]Processing daily minimum temperature data...[/blue]")
        
        # Convert units from Kelvin to Celsius
        tasmin_data = convert_units(data, "K", "C")
        
        # Standardize coordinates
        tasmin_data = self._standardize_coordinates(tasmin_data)
        
        # Choose processing strategy
        strategy = self._select_processing_strategy(gdf, chunk_by_county)
        
        # Process the data (no threshold needed for tasmin)
        return strategy.process(
            data=tasmin_data,
            gdf=gdf,
            variable='tasmin',
            scenario=scenario,
            threshold=0.0,  # Not used for tasmin
            n_workers=self.n_workers
        )
    
    def _select_processing_strategy(self, gdf: gpd.GeoDataFrame, chunk_by_county: bool):
        """Select processing strategy for daily minimum temperature data.
        
        Args:
            gdf: County geometries
            chunk_by_county: Whether chunked processing is requested
            
        Returns:
            Selected processing strategy instance
        """
        if len(gdf) > 50 or not chunk_by_county:
            console.print("[cyan]🚀 Using ultra-fast processing (best performance)[/cyan]")
            return UltraFastStrategy()
        else:
            console.print("[cyan]Using vectorized processing[/cyan]")
            return VectorizedStrategy()
    
    def process_zarr_file(
        self,
        zarr_path: Path,
        gdf: gpd.GeoDataFrame,
        scenario: str = 'historical',
        chunk_by_county: bool = True
    ) -> pd.DataFrame:
        """Process a Zarr file containing daily minimum temperature data.
        
        Args:
            zarr_path: Path to Zarr dataset
            gdf: County geometries
            scenario: Scenario name
            chunk_by_county: Whether to use chunked processing
            
        Returns:
            DataFrame with daily minimum temperature statistics
        """
        console.print(f"[blue]Opening daily minimum temperature Zarr dataset:[/blue] {zarr_path}")
        
        # Open with optimizations
        ds = xr.open_zarr(zarr_path, chunks={'time': 365})
        
        # Get daily minimum temperature data
        if 'tasmin' not in ds.data_vars:
            raise ValueError("Daily minimum temperature variable 'tasmin' not found in dataset")
        
        tasmin_data = ds['tasmin']
        
        return self.process_variable_data(
            data=tasmin_data,
            gdf=gdf,
            scenario=scenario,
            chunk_by_county=chunk_by_county
        ) 