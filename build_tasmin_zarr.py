#!/usr/bin/env python
"""
Script to build Zarr stores from tasmin (minimum temperature) NetCDF files.
This script processes daily minimum temperature data and organizes it by region
in the same structure as other climate variables.
"""

import sys
import xarray as xr
import numpy as np
from pathlib import Path
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from climate_zarr.utils.data_utils import convert_units
from climate_zarr.climate_config import get_config

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TasminZarrBuilder:
    """Builds Zarr stores from tasmin NetCDF files."""
    
    def __init__(self, input_dir: Path, output_dir: Path, scenario: str = "ssp370"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.scenario = scenario
        self.config = get_config()
        
        # Define regions and their coordinate bounds
        self.regions = {
            'conus': {'lat': (24, 50), 'lon': (-125, -66)},
            'alaska': {'lat': (52, 72), 'lon': (-169, -129)},
            'hawaii': {'lat': (18, 23), 'lon': (-161, -154)},
            'puerto_rico': {'lat': (17, 19), 'lon': (-68, -65)},
            'guam': {'lat': (13, 14), 'lon': (144, 145)}
        }
    
    def _convert_longitude(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert longitude coordinates to -180 to 180 range if needed."""
        if np.any(ds.lon > 180):
            ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
            ds = ds.sortby('lon')
        return ds
    
    def _get_region_slice(self, ds: xr.Dataset, region: str) -> xr.Dataset:
        """Extract data for a specific region."""
        bounds = self.regions[region]
        
        # Handle the case where longitude wraps around 180/-180
        if bounds['lon'][0] > bounds['lon'][1]:  # e.g., for Alaska
            mask = (ds.lon >= bounds['lon'][0]) | (ds.lon <= bounds['lon'][1])
            ds_region = ds.where(mask, drop=True)
        else:
            ds_region = ds.sel(
                lon=slice(bounds['lon'][0], bounds['lon'][1]),
                lat=slice(bounds['lat'][0], bounds['lat'][1])
            )
        
        return ds_region
    
    def process_file(self, file_path: Path, region: str, output_store: str):
        """Process a single NetCDF file and add it to the Zarr store."""
        try:
            # Read the NetCDF file
            ds = xr.open_dataset(file_path)
            
            # Convert longitude range if needed
            ds = self._convert_longitude(ds)
            
            # Extract region data
            ds_region = self._get_region_slice(ds, region)
            
            # Convert units if needed (K to °C)
            if 'units' in ds_region.tasmin.attrs and ds_region.tasmin.attrs['units'] in ['K', 'Kelvin']:
                ds_region['tasmin'] = ds_region['tasmin'] - 273.15
                ds_region.tasmin.attrs['units'] = 'C'
            
            # Write to Zarr store
            ds_region.to_zarr(output_store, append_dim='time')
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return False
    
    def build_zarr_stores(self):
        """Build Zarr stores for all regions."""
        console.print("[bold green]🚀 Starting tasmin Zarr store creation[/bold green]")
        
        # Get list of input files
        nc_files = sorted(self.input_dir.glob(f"tasmin_day_*_{self.scenario}_*.nc"))
        if not nc_files:
            console.print(f"[red]❌ No NetCDF files found in {self.input_dir}[/red]")
            return
        
        # Process each region
        for region in self.regions:
            console.print(f"\n[bold blue]Processing region: {region.upper()}[/bold blue]")
            
            # Create output directory structure
            output_store = self.output_dir / "tasmin" / region / self.scenario / f"{region}_{self.scenario}_tasmin_daily.zarr"
            output_store.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize Zarr store with first file to set up structure
            first_file = nc_files[0]
            try:
                ds = xr.open_dataset(first_file)
                ds = self._convert_longitude(ds)
                ds_region = self._get_region_slice(ds, region)
                
                if 'units' in ds_region.tasmin.attrs and ds_region.tasmin.attrs['units'] in ['K', 'Kelvin']:
                    ds_region['tasmin'] = ds_region['tasmin'] - 273.15
                    ds_region.tasmin.attrs['units'] = 'C'
                
                # Initialize store
                ds_region.to_zarr(output_store, mode='w')
                console.print(f"[green]✅ Initialized Zarr store for {region}[/green]")
                
            except Exception as e:
                console.print(f"[red]❌ Failed to initialize {region}: {str(e)}[/red]")
                continue
            
            # Process remaining files
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"Processing {region}...", total=len(nc_files))
                
                for nc_file in nc_files[1:]:
                    progress.update(task, advance=1)
                    if not self.process_file(nc_file, region, output_store):
                        console.print(f"[yellow]⚠️ Warning: Failed to process {nc_file.name}[/yellow]")
                
            console.print(f"[green]✅ Completed {region}[/green]")
        
        console.print("\n[bold green]🎉 Zarr store creation complete![/bold green]")

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build Zarr stores from tasmin NetCDF files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/tasmin/ssp370",
        help="Directory containing tasmin NetCDF files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="climate_outputs/zarr",
        help="Base directory for output Zarr stores"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="ssp370",
        help="Climate scenario (default: ssp370)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        console.print(f"[red]❌ Input directory not found: {input_dir}[/red]")
        sys.exit(1)
    
    # Create builder and run
    builder = TasminZarrBuilder(
        input_dir=input_dir,
        output_dir=args.output_dir,
        scenario=args.scenario
    )
    
    builder.build_zarr_stores()

if __name__ == "__main__":
    main()