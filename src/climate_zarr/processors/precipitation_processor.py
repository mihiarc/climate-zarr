#!/usr/bin/env python
"""Precipitation data processor for county-level statistics."""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from typing import List, Dict, Any, Optional
from pathlib import Path
from rich.console import Console

from .base_processor import BaseCountyProcessor
from .strategies import UltraFastStrategy
from ..utils.data_utils import convert_units, calculate_statistics
from ..utils.spatial_utils import get_coordinate_arrays

console = Console()


class PrecipitationProcessor(BaseCountyProcessor):
    """Processor for precipitation data with spatial interpolation and zonal statistics support."""
    
    def __init__(self, n_workers: int = 4, strategy: str = 'zonal_statistics'):
        """
        Initialize the precipitation processor.
        
        Args:
            n_workers: Number of worker processes
            strategy: Processing strategy ('ultra_fast', 'vectorized', 'comprehensive', 'interpolation', 'zonal_statistics')
        """
        super().__init__(n_workers)
        self.strategy_name = strategy
        self.strategy = None
        self.variable = 'pr'
        self.units = 'mm/day'
    
    def process_variable_data(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_mm: float = 25.4,
        chunk_by_county: bool = True,
        zarr_path: Optional[Path] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Process precipitation data for all counties.
        
        Args:
            data: Precipitation data array
            gdf: County geometries
            scenario: Scenario name
            threshold_mm: Precipitation threshold in mm/day
            chunk_by_county: Whether to use chunked processing
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with precipitation statistics
        """
        console.print("[blue]Processing precipitation data...[/blue]")
        
        # Standardize the GeoDataFrame columns (including adding raster_id)
        standardized_gdf = self._standardize_columns(gdf)
        
        # Convert units
        pr_data = convert_units(data, data.attrs.get('units', 'kg/m²/s'), self.units)
        
        # Standardize coordinates
        pr_data = self._standardize_coordinates(pr_data)
        
        # Choose processing strategy
        strategy = self._select_processing_strategy(pr_data, standardized_gdf, chunk_by_county)
        
        # Process the data
        return strategy.process(
            data=pr_data,
            gdf=standardized_gdf,
            variable='pr',
            scenario=scenario,
            threshold=threshold_mm,
            n_workers=self.n_workers,
            zarr_path=zarr_path
        )
    
    def _select_processing_strategy(self, data: xr.DataArray, gdf: gpd.GeoDataFrame, chunk_by_county: bool):
        """Return the standardized fastest processing strategy (UltraFast)."""
        console.print("[cyan]🚀 Using ultra-fast processing (standardized default)[/cyan]")
        return UltraFastStrategy()
        
    def _get_strategy(self):
        """Get the processing strategy instance (always UltraFast after refactoring)."""
        if self.strategy is None:
            console.print("[cyan]🚀 Using ultra-fast processing (only available strategy)[/cyan]")
            self.strategy = UltraFastStrategy()
        return self.strategy
    
    def process_counties(
        self,
        shapefile_path: str,
        zarr_path: str,
        years: List[int],
        output_path: Optional[str] = None,
        statistic: str = 'mean'
    ) -> pd.DataFrame:
        """
        Process counties with precipitation data using spatial interpolation.
        
        Args:
            shapefile_path: Path to county shapefile
            zarr_path: Path to zarr store
            years: List of years to process
            output_path: Optional output CSV path
            statistic: Statistic to calculate ('mean', 'sum', etc.)
            
        Returns:
            DataFrame with county statistics
        """
        console.print(f"[cyan]🌧️  Processing precipitation data using {self.strategy_name} strategy[/cyan]")
        
        # Load and prepare data
        gdf = self.prepare_shapefile(shapefile_path)
        zarr_ds = xr.open_zarr(zarr_path)
        
        # Get precipitation data
        pr_data = zarr_ds[self.variable]
        
        # Convert units
        console.print(f"[cyan]Converting precipitation units from {pr_data.attrs.get('units', 'unknown')} to {self.units}[/cyan]")
        pr_data_converted = convert_units(pr_data, pr_data.attrs.get('units', 'kg/m²/s'), self.units)
        
        # Standardize coordinates
        pr_data_standardized = self._standardize_coordinates(pr_data_converted)
        
        # Get processing strategy
        strategy = self._get_strategy()
        
        # Process counties
        if hasattr(strategy, 'process_counties'):
            # New interpolation strategy
            results, stats = strategy.process_counties(
                gdf, pr_data_standardized, years, statistic
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Print processing statistics
            console.print(f"[green]✅ Processing completed![/green]")
            console.print(f"[cyan]📊 Processing Statistics:[/cyan]")
            console.print(f"  • Total counties: {stats['total_counties']}")
            console.print(f"  • Total records: {stats['total_records']}")
            console.print(f"  • Valid records: {stats['valid_records']}")
            console.print(f"  • Coverage rate: {stats['coverage_rate']:.1%}")
            console.print(f"  • Rasterized counties: {stats['rasterized_counties']}")
            console.print(f"  • Interpolated counties: {stats['interpolated_counties']}")
            
            console.print(f"[cyan]📈 Extraction Methods:[/cyan]")
            for method, count in stats['extraction_methods'].items():
                if count > 0:
                    console.print(f"  • {method}: {count} records")
            
        else:
            # Legacy strategy
            df = strategy.process_counties(gdf, pr_data_standardized, years, statistic)
        
        # Save results
        if output_path:
            df.to_csv(output_path, index=False)
            console.print(f"[green]💾 Results saved to: {output_path}[/green]")
        
        return df 