#!/usr/bin/env python
"""Precipitation data processor for county-level statistics."""

from pathlib import Path
from typing import Dict
import pandas as pd
import geopandas as gpd
import xarray as xr
from rich.console import Console

from .base_processor import BaseCountyProcessor
from .processing_strategies import VectorizedStrategy, UltraFastStrategy
from ..utils.data_utils import convert_units

console = Console()


class PrecipitationProcessor(BaseCountyProcessor):
    """Processor for precipitation data (pr variable)."""
    
    def process_variable_data(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_mm: float = 25.4, # 1 inch
        chunk_by_county: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Process precipitation data for all counties.
        
        Args:
            data: Precipitation data array
            gdf: County geometries
            scenario: Scenario name
            threshold_mm: Precipitation threshold in mm
            chunk_by_county: Whether to use chunked processing
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with precipitation statistics
        """
        console.print("[blue]Processing precipitation data...[/blue]")
        
        # Convert units from kg/m²/s to mm/day
        pr_data = convert_units(data, "kg/m2/s", "mm/day")
        
        # Standardize coordinates
        pr_data = self._standardize_coordinates(pr_data)
        
        # Choose processing strategy based on dataset size and parameters
        strategy = self._select_processing_strategy(pr_data, gdf, chunk_by_county)
        
        # Process the data
        return strategy.process(
            data=pr_data,
            gdf=gdf,
            variable='pr',
            scenario=scenario,
            threshold=threshold_mm,
            n_workers=self.n_workers
        )
    
    def _select_processing_strategy(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        chunk_by_county: bool
    ):
        """Select the best processing strategy based on data characteristics.
        
        Args:
            data: Climate data array
            gdf: County geometries
            chunk_by_county: Whether chunked processing is requested
            
        Returns:
            Selected processing strategy instance
        """
        # For regional datasets or when ultra-fast processing is preferred, use ultra-fast
        if len(gdf) > 50 or not chunk_by_county:
            console.print("[cyan]🚀 Using ultra-fast processing (best performance)[/cyan]")
            return UltraFastStrategy()
        
        # Default to vectorized processing for smaller datasets
        else:
            console.print("[cyan]Using vectorized processing[/cyan]")
            return VectorizedStrategy()
    
    def process_zarr_file(
        self,
        zarr_path: Path,
        gdf: gpd.GeoDataFrame,
        scenario: str = 'historical',
        threshold_mm: float = 25.4,
        chunk_by_county: bool = True
    ) -> pd.DataFrame:
        """Process a Zarr file containing precipitation data.
        
        Args:
            zarr_path: Path to Zarr dataset
            gdf: County geometries
            scenario: Scenario name
            threshold_mm: Precipitation threshold in mm
            chunk_by_county: Whether to use chunked processing
            
        Returns:
            DataFrame with precipitation statistics
        """
        console.print(f"[blue]Opening precipitation Zarr dataset:[/blue] {zarr_path}")
        
        # Open with optimizations
        ds = xr.open_zarr(zarr_path, chunks={'time': 365})
        
        # Get precipitation data
        if 'pr' not in ds.data_vars:
            raise ValueError("Precipitation variable 'pr' not found in dataset")
        
        pr_data = ds['pr']
        
        return self.process_variable_data(
            data=pr_data,
            gdf=gdf,
            scenario=scenario,
            threshold_mm=threshold_mm,
            chunk_by_county=chunk_by_county
        ) 