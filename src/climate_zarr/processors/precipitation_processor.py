#!/usr/bin/env python
"""Precipitation data processor for county-level statistics."""

from pathlib import Path
from typing import Dict
import pandas as pd
import geopandas as gpd
import xarray as xr
from rich.console import Console

from .base_processor import BaseCountyProcessor
from .processing_strategies import VectorizedStrategy
from .strategy_config import create_processing_plan
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
        **kwargs
    ) -> pd.DataFrame:
        """Process precipitation data for all counties.
        
        Args:
            data: Precipitation data array
            gdf: County geometries
            scenario: Scenario name
            threshold_mm: Precipitation threshold in mm
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with precipitation statistics
        """
        console.print("[blue]Processing precipitation data...[/blue]")
        
        # Convert units from kg/m²/s to mm/day
        pr_data = convert_units(data, "kg/m2/s", "mm/day")
        
        # Standardize coordinates
        pr_data = self._standardize_coordinates(pr_data)
        
        # Create intelligent processing plan with optimal strategy selection
        console.print("[blue]Creating optimized processing plan...[/blue]")
        processing_plan = create_processing_plan(
            data=pr_data,
            gdf=gdf,
            variable='pr',
            scenario=scenario
        )
        
        # Execute processing with selected strategy
        strategy = processing_plan['strategy']
        return strategy.process(
            data=pr_data,
            gdf=gdf,
            variable='pr',
            scenario=scenario,
            threshold=threshold_mm,
            n_workers=processing_plan['config'].parallel_workers
        )
    
    
    def process_zarr_file(
        self,
        zarr_path: Path,
        gdf: gpd.GeoDataFrame,
        scenario: str = 'historical',
        threshold_mm: float = 25.4
    ) -> pd.DataFrame:
        """Process a Zarr file containing precipitation data.
        
        Args:
            zarr_path: Path to Zarr dataset
            gdf: County geometries
            scenario: Scenario name
            threshold_mm: Precipitation threshold in mm
            
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
            threshold_mm=threshold_mm
        ) 