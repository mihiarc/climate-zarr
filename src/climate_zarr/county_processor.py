#!/usr/bin/env python
"""Modern county processor that replaces the monolithic implementation."""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import geopandas as gpd
import xarray as xr
from rich.console import Console

from .processors import (
    PrecipitationProcessor,
    TemperatureProcessor,
    TasMaxProcessor,
    TasMinProcessor
)
from .utils.output_utils import get_output_manager

console = Console()


class ModernCountyProcessor:
    """Modern, modular county processor for climate data analysis.
    
    This class replaces the monolithic implementation with a clean,
    modular architecture that delegates to specialized processors.
    """
    
    def __init__(
        self,
        n_workers: int = 4
    ):
        """Initialize the modern county processor.
        
        Args:
            n_workers: Number of worker processes
        """
        self.n_workers = n_workers
        
        # Initialize variable-specific processors
        self._processors = {
            'pr': PrecipitationProcessor(n_workers),
            'tas': TemperatureProcessor(n_workers),
            'tasmax': TasMaxProcessor(n_workers),
            'tasmin': TasMinProcessor(n_workers),
        }
    
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
        # Use the first processor's method (they all have the same implementation)
        return self._processors['pr'].prepare_shapefile(shapefile_path, target_crs)
    
    def process_zarr_data(
        self,
        zarr_path: Path,
        gdf: gpd.GeoDataFrame,
        scenario: str = 'historical',
        variable: str = 'pr',
        threshold: float = 25.4
    ) -> pd.DataFrame:
        """Process Zarr data using the appropriate variable processor.
        
        Args:
            zarr_path: Path to Zarr dataset
            gdf: County geometries
            scenario: Scenario name
            variable: Climate variable to process
            threshold: Threshold value for the variable
            
        Returns:
            DataFrame with processed results
        """
        console.print(f"[blue]Opening Zarr dataset:[/blue] {zarr_path}")
        
        # Validate variable
        if variable not in self._processors:
            raise ValueError(f"Unsupported variable: {variable}. Supported: {list(self._processors.keys())}")
        
        # Get the appropriate processor
        processor = self._processors[variable]
        
        # Open the dataset
        ds = xr.open_zarr(zarr_path, chunks={'time': 365})
        
        # Check if variable exists in dataset
        if variable not in ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset. Available: {list(ds.data_vars)}")
        
        # Get the data array
        data = ds[variable]
        
        # Process based on variable type
        if variable == 'pr':
            return processor.process_variable_data(
                data=data,
                gdf=gdf,
                scenario=scenario,
                threshold_mm=threshold
            )
        elif variable == 'tas':
            return processor.process_variable_data(
                data=data,
                gdf=gdf,
                scenario=scenario
            )
        elif variable == 'tasmax':
            return processor.process_variable_data(
                data=data,
                gdf=gdf,
                scenario=scenario,
                threshold_temp_c=threshold
            )
        elif variable == 'tasmin':
            return processor.process_variable_data(
                data=data,
                gdf=gdf,
                scenario=scenario
            )
    
    def get_processor(self, variable: str):
        """Get the processor for a specific variable.
        
        Args:
            variable: Climate variable name
            
        Returns:
            Processor instance for the variable
        """
        if variable not in self._processors:
            raise ValueError(f"Unsupported variable: {variable}")
        return self._processors[variable]
    
    def save_results(
        self,
        results_df: pd.DataFrame,
        variable: str,
        region: str,
        scenario: str = "historical",
        threshold: Optional[float] = None,
        output_path: Optional[Path] = None,
        metadata: Optional[Dict] = None
    ) -> Path:
        """Save results using standardized output management.
        
        Args:
            results_df: Results DataFrame to save
            variable: Climate variable processed
            region: Region name
            scenario: Scenario name
            threshold: Threshold value used
            output_path: Custom output path (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Path where results were saved
        """
        output_manager = get_output_manager()
        
        if output_path is None:
            output_path = output_manager.get_output_path(
                variable=variable,
                region=region,
                scenario=scenario,
                threshold=threshold
            )
        
        # Prepare metadata
        save_metadata = {
            "processing_info": {
                "variable": variable,
                "region": region,
                "scenario": scenario,
                "threshold": threshold,
                "n_workers": self.n_workers
            },
            "data_summary": {
                "counties_processed": len(results_df['county_id'].unique()) if 'county_id' in results_df.columns else len(results_df),
                "years_processed": len(results_df['year'].unique()) if 'year' in results_df.columns else "unknown",
                "total_records": len(results_df)
            }
        }
        
        if metadata:
            save_metadata.update(metadata)
        
        # Save with metadata
        return output_manager.save_with_metadata(
            data=results_df,
            output_path=output_path,
            metadata=save_metadata,
            save_method="csv"
        )
    
    def close(self):
        """Clean up resources for all processors."""
        for processor in self._processors.values():
            processor.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close() 