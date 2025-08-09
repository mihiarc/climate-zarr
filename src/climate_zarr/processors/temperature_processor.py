#!/usr/bin/env python
"""Temperature data processor for county-level statistics."""

import pandas as pd
import geopandas as gpd
import xarray as xr
from rich.console import Console

from typing import Optional
from pathlib import Path
from .base_processor import BaseCountyProcessor
from .strategies import UltraFastStrategy
from ..utils.data_utils import convert_units

console = Console()


class TemperatureProcessor(BaseCountyProcessor):
    """Processor for mean temperature data (tas variable)."""
    
    def process_variable_data(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        chunk_by_county: bool = True,
        zarr_path: Optional[Path] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Process temperature data for all counties.
        
        Args:
            data: Temperature data array
            gdf: County geometries
            scenario: Scenario name
            chunk_by_county: Whether to use chunked processing
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with temperature statistics
        """
        console.print("[blue]Processing mean temperature data...[/blue]")
        
        # Convert units from Kelvin to Celsius
        tas_data = convert_units(data, "K", "C")
        
        # Standardize coordinates
        tas_data = self._standardize_coordinates(tas_data)
        
        # Choose processing strategy
        strategy = self._select_processing_strategy(gdf, chunk_by_county)
        
        # Process the data (no threshold needed for temperature)
        return strategy.process(
            data=tas_data,
            gdf=gdf,
            variable='tas',
            scenario=scenario,
            threshold=0.0,  # Not used for temperature
            n_workers=self.n_workers,
            zarr_path=zarr_path
        )
    
    def _select_processing_strategy(self, gdf: gpd.GeoDataFrame, chunk_by_county: bool):
        """Return the standardized fastest processing strategy (UltraFast)."""
        console.print("[cyan]🚀 Using ultra-fast processing (standardized default)[/cyan]")
        return UltraFastStrategy()
    
    def process_zarr_file(
        self,
        zarr_path: Path,
        gdf: gpd.GeoDataFrame,
        scenario: str = 'historical',
        chunk_by_county: bool = True
    ) -> pd.DataFrame:
        """Process a Zarr file containing temperature data.
        
        Args:
            zarr_path: Path to Zarr dataset
            gdf: County geometries
            scenario: Scenario name
            chunk_by_county: Whether to use chunked processing
            
        Returns:
            DataFrame with temperature statistics
        """
        console.print(f"[blue]Opening temperature Zarr dataset:[/blue] {zarr_path}")
        
        # Open with optimizations
        ds = xr.open_zarr(zarr_path, chunks={'time': 365})
        
        # Get temperature data
        if 'tas' not in ds.data_vars:
            raise ValueError("Temperature variable 'tas' not found in dataset")
        
        tas_data = ds['tas']
        
        return self.process_variable_data(
            data=tas_data,
            gdf=gdf,
            scenario=scenario,
            chunk_by_county=chunk_by_county
        ) 