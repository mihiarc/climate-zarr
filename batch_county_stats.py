#!/usr/bin/env python
"""
Batch script to calculate county statistics from Zarr climate files
using the modern modular architecture from src/climate_zarr/

Optimized version with parallel processing and vectorized operations.
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import os
import argparse
from rich.console import Console
# Progress components available if needed for future enhancements
# from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from climate_zarr.county_processor import ModernCountyProcessor
from climate_zarr.climate_config import get_config

console = Console()


class OptimizedBatchCountyProcessor:
    """Optimized batch processor for generating county statistics from Zarr files."""
    
    def __init__(self, zarr_base_dir: Path, regions: List[str], scenario: str = "ssp370", max_workers: int = 32):
        self.zarr_base_dir = Path(zarr_base_dir)
        self.regions = regions
        self.scenario = scenario
        self.config = get_config()
        # Default to using all available CPUs for county-level multiprocessing
        self.processor = ModernCountyProcessor(n_workers=os.cpu_count() or 4)
        self.max_workers = max_workers if max_workers > 0 else (os.cpu_count() or 4)
        self._lock = threading.Lock()  # For thread-safe console output
        
    def get_shapefile_for_region(self, region: str) -> Path:
        """Get the appropriate shapefile path for a region."""
        region_files = {
            'conus': 'conus_counties.shp',
            'alaska': 'alaska_counties.shp', 
            'hawaii': 'hawaii_counties.shp',
            'guam': 'guam_counties.shp',
            'puerto_rico': 'puerto_rico_counties.shp',
        }
        
        shapefile_name = region_files.get(region, f'{region}_counties.shp')
        shapefile_path = Path('regional_counties') / shapefile_name
        
        if not shapefile_path.exists():
            console.print(f"[red]❌ Shapefile not found: {shapefile_path}[/red]")
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
        
        return shapefile_path
    
    def find_zarr_files(self, region: str) -> Dict[str, Path]:
        """Find Zarr files for a given region."""
        zarr_files = {}
        variables = ['pr', 'tas', 'tasmax', 'tasmin']
        
        for variable in variables:
            zarr_path = self.zarr_base_dir / variable / region / self.scenario / f"{region}_{self.scenario}_{variable}_daily.zarr"
            if zarr_path.exists():
                zarr_files[variable] = zarr_path
        
        return zarr_files
    
    def process_variable(self, zarr_path: Path, gdf, variable: str, n_workers: Optional[int] = None) -> pd.DataFrame:
        """Process a climate variable using modern processors."""
        # Create a local processor so we can control per-region workers precisely
        local_processor = ModernCountyProcessor(n_workers=n_workers or self.processor.n_workers)
        return local_processor.process_zarr_data(
            zarr_path=zarr_path,
            gdf=gdf, 
            scenario=self.scenario,
            variable=variable,
            # Use UltraFast strategy (county-level multiprocessing inside the processor)
            chunk_by_county=False
        )
    
    def process_region(self, region: str, n_workers: Optional[int] = None) -> pd.DataFrame:
        """Process all variables for a specific region using modern processors."""
        # Find Zarr files
        zarr_files = self.find_zarr_files(region)
        if not zarr_files:
            return pd.DataFrame()
        
        # Load and prepare shapefile
        shapefile_path = self.get_shapefile_for_region(region)
        # Use a lightweight ModernCountyProcessor for shapefile preparation only
        prep_processor = ModernCountyProcessor(n_workers=1)
        gdf = prep_processor.prepare_shapefile(shapefile_path)
        
        # Process each variable using modern processors
        variable_results = {}
        
        for variable, zarr_path in zarr_files.items():
            try:
                results_df = self.process_variable(zarr_path, gdf, variable, n_workers=n_workers)
                if not results_df.empty:
                    variable_results[variable] = results_df
            except Exception as e:
                # Log error but continue with other variables
                console.print(f"[red]❌ {region}/{variable}: {str(e)}[/red]")
                continue
        
        # Combine variable results into final format
        return self._combine_variable_results(variable_results) if variable_results else pd.DataFrame()
    
    def _combine_variable_results(self, variable_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine results from multiple variables into final format using efficient pandas operations."""
        if not variable_results:
            return pd.DataFrame()
        
        # Start with the first variable as base
        base_var = list(variable_results.keys())[0]
        result_df = variable_results[base_var].copy()
        
        # Create base columns
        result_df = result_df.rename(columns={'county_id': 'cid2'})
        result_df['name'] = result_df['county_name'] + ', ' + result_df['state']
        # scenario is not part of the final output format; omit from final CSV
        
        # Initialize all output columns with defaults
        output_columns = {
            'annual_mean_temp': 0.0,
            'annual_total_precip': 0.0,
            'daysabove1in': 0,
            'daysabove90F': 0,
            'daysbelow32F': 0,
            'tmaxavg': 0.0
        }
        
        for col, default_val in output_columns.items():
            result_df[col] = default_val
        
        # Map processor output columns to final columns
        variable_mapping = {
            'pr': {'days_above_threshold': 'daysabove1in', 'total_annual_precip_mm': 'annual_total_precip'},
            'tas': {'mean_annual_temp_c': 'annual_mean_temp'},
            'tasmax': {'heat_index_days': 'daysabove90F', 'mean_annual_tasmax_c': 'tmaxavg'},
            'tasmin': {'cold_days': 'daysbelow32F'}
        }
        
        # Use vectorized operations to merge data from each variable
        for variable, df in variable_results.items():
            if variable in variable_mapping:
                # Merge on county_id and year
                merge_df = df[['county_id', 'year'] + list(variable_mapping[variable].keys())].copy()
                
                for source_col, target_col in variable_mapping[variable].items():
                    if source_col in merge_df.columns:
                        # Use pandas merge for efficient column updating
                        temp_merge = result_df[['cid2', 'year']].merge(
                            merge_df[['county_id', 'year', source_col]], 
                            left_on=['cid2', 'year'], 
                            right_on=['county_id', 'year'], 
                            how='left'
                        )
                        result_df[target_col] = temp_merge[source_col].fillna(result_df[target_col])
        
        # Select final columns
        # Conform to climate_output_format.yaml
        # Include tasmin-derived freezing days if available
        final_columns = ['cid2', 'year', 'name', 'daysabove1in', 'daysabove90F', 'tmaxavg', 'annual_mean_temp', 'annual_total_precip', 'daysbelow32F']
        result_df = result_df[final_columns].drop_duplicates(['cid2', 'year']).reset_index(drop=True)
        return result_df
    
    def _process_region_wrapper(self, region: str, progress_callback=None) -> Optional[pd.DataFrame]:
        """Thread-safe wrapper for processing a single region."""
        try:
            result = self.process_region(region)
            if progress_callback:
                progress_callback(region, len(result) if not result.empty else 0)
            return result if not result.empty else None
        except Exception as e:
            with self._lock:
                console.print(f"[red]❌ Error processing {region}: {e}[/red]")
            return None
    
    def run_batch_processing(self, output_file: str, use_parallel: bool = True) -> pd.DataFrame:
        """Run batch processing for all regions using multiprocessing with Zarr streaming."""
        console.print("[bold green]🚀 Processing climate county statistics[/bold green]")
        console.print(f"[cyan]Regions: {', '.join(self.regions)} | Scenario: {self.scenario} | Mode: {'Multiprocessing' if use_parallel else 'Sequential'}[/cyan]")
        
        all_data = []
        total_cpus = os.cpu_count() or 4
        if use_parallel and len(self.regions) > 1:
            # Plan resources: avoid oversubscription by splitting CPUs across regions
            region_pool_size = min(len(self.regions), self.max_workers, total_cpus)
            per_region_workers = max(1, total_cpus // region_pool_size)
            console.print(f"[cyan]🚀 Region pool: {region_pool_size} | County workers/region: {per_region_workers}[/cyan]")
            with ProcessPoolExecutor(max_workers=region_pool_size) as executor:
                futures = {
                    executor.submit(self._process_region_standalone, region, self.zarr_base_dir, self.scenario, per_region_workers): region
                    for region in self.regions
                }
                for future in as_completed(futures):
                    region = futures[future]
                    try:
                        result = future.result()
                        if not result.empty:
                            all_data.append(result)
                            console.print(f"[green]✅ {region.upper()}: {len(result)} records[/green]")
                        else:
                            console.print(f"[yellow]⚠️ {region.upper()}: No data[/yellow]")
                    except Exception as e:
                        console.print(f"[red]❌ {region.upper()}: {str(e)}[/red]")
        else:
            # Sequential processing
            per_region_workers = min(self.max_workers, total_cpus)
            for region in self.regions:
                try:
                    region_data = self.process_region(region, n_workers=per_region_workers)
                    if not region_data.empty:
                        all_data.append(region_data)
                        console.print(f"[green]✅ {region.upper()}: {len(region_data)} records[/green]")
                    else:
                        console.print(f"[yellow]⚠️ {region.upper()}: No data[/yellow]")
                except Exception as e:
                    console.print(f"[red]❌ {region.upper()}: {str(e)}[/red]")
        
        # Combine and finalize results
        if all_data:
            # Use efficient concatenation
            final_df = pd.concat(all_data, ignore_index=True, sort=False)
            
            # Ensure proper column order and types
            columns = ['cid2', 'year', 'name', 'daysabove1in', 'daysabove90F', 'tmaxavg', 'annual_mean_temp', 'annual_total_precip']
            final_df = final_df.reindex(columns=columns, fill_value=0.0)
            
            # Optimize data types for memory efficiency
            dtype_map = {
                'cid2': 'int32', 'year': 'int16',
                'annual_mean_temp': 'float32', 'annual_total_precip': 'float32',
                'daysabove1in': 'int16', 'daysabove90F': 'int16', 'tmaxavg': 'float32'
            }
            final_df = final_df.astype(dtype_map).sort_values(['cid2', 'year']).reset_index(drop=True)
            
            # Save results
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_df.to_csv(output_path, index=False, float_format='%.2f')
            
            # Summary
            console.print(f"\n[bold green]✅ Complete: {len(final_df):,} records, {final_df['cid2'].nunique():,} counties, {final_df['year'].nunique()} years[/bold green]")
            console.print(f"[cyan]Saved: {output_path} ({final_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB)[/cyan]")
            
            return final_df
        else:
            console.print("[red]❌ No data processed[/red]")
            return pd.DataFrame()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.processor.close()
    
    @staticmethod
    def _process_region_standalone(region: str, zarr_base_dir: Path, scenario: str, per_region_workers: int = 1) -> pd.DataFrame:
        """Standalone region processing function for multiprocessing.
        
        This function recreates the processor in each process to avoid
        pickling issues and enables independent Zarr streaming access.
        """
        # Create fresh processor instance in each process
        processor = ModernCountyProcessor(n_workers=per_region_workers)
        
        try:
            # Create temporary batch processor instance
            temp_batch = OptimizedBatchCountyProcessor(
                zarr_base_dir=zarr_base_dir,
                regions=[region],  # Single region
                scenario=scenario,
                max_workers=per_region_workers
            )
            
            # Process the region using streaming Zarr access
            return temp_batch.process_region(region, n_workers=per_region_workers)
            
        finally:
            processor.close()


def main():
    """Main entry point for the optimized batch processing script."""
    parser = argparse.ArgumentParser(
        description="Generate county climate statistics from Zarr files using optimized processing"
    )
    parser.add_argument(
        "--zarr-dir", 
        type=str, 
        default="climate_outputs/zarr",
        help="Base directory containing Zarr files (default: climate_outputs/zarr)"
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="conus,alaska,hawaii,guam,puerto_rico",
        help="Comma-separated list of regions to process (default: all regions)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="ssp370",
        help="Climate scenario to process (default: ssp370)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="climate_county_stats_optimized.csv",
        help="Output CSV file path (default: climate_county_stats_optimized.csv)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel region processing (default: sequential)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum number of parallel workers (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Parse regions
    regions = [r.strip() for r in args.regions.split(',')]
    
    # Validate zarr directory
    zarr_dir = Path(args.zarr_dir)
    if not zarr_dir.exists():
        console.print(f"[red]❌ Zarr directory not found: {zarr_dir}[/red]")
        sys.exit(1)
    
    # Create processor and run
    with OptimizedBatchCountyProcessor(
        zarr_base_dir=zarr_dir,
        regions=regions,
        scenario=args.scenario,
        max_workers=args.max_workers
    ) as processor:
        result_df = processor.run_batch_processing(args.output, use_parallel=args.parallel)
        
        if not result_df.empty:
            # Display sample of results
            console.print("\n[bold cyan]📊 Sample of generated data:[/bold cyan]")
            console.print(result_df.head(10).to_string(index=False))
            
            # Display summary statistics
            console.print("\n[bold cyan]📈 Summary statistics:[/bold cyan]")
            summary_stats = result_df.describe()
            console.print(summary_stats.to_string())
            
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()