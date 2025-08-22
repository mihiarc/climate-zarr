#!/usr/bin/env python
"""Simplified CLI for county statistics using the modular processor."""

import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

from climate_zarr.county_processor import ModernCountyProcessor
from climate_zarr.utils.output_utils import get_output_manager

console = Console()


def main():
    """Main function with simplified CLI."""
    parser = argparse.ArgumentParser(
        description="Calculate county statistics using modular architecture"
    )
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to Zarr dataset"
    )
    parser.add_argument(
        "shapefile_path", 
        type=Path,
        help="Path to county shapefile"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output CSV file (default: auto-generated based on parameters)"
    )
    parser.add_argument(
        "-s", "--scenario",
        type=str,
        default="historical",
        help="Scenario name (default: historical)"
    )
    parser.add_argument(
        "-v", "--variable",
        type=str,
        default="pr",
        choices=["pr", "tas", "tasmax", "tasmin"],
        help="Variable to process (default: pr)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=25.4,
        help="Threshold value (default: 25.4)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.zarr_path.exists():
        console.print(f"[red]Zarr path does not exist: {args.zarr_path}[/red]")
        return
    
    if not args.shapefile_path.exists():
        console.print(f"[red]Shapefile does not exist: {args.shapefile_path}[/red]")
        return
    
    # Set up output manager and determine output path
    output_manager = get_output_manager()
    
    if args.output is None:
        # Auto-generate output path based on parameters
        # Try to infer region from shapefile name
        region = "unknown"
        shapefile_name = args.shapefile_path.stem.lower()
        for region_name in ["conus", "alaska", "hawaii", "puerto_rico", "guam"]:
            if region_name in shapefile_name:
                region = region_name
                break
        
        args.output = output_manager.get_output_path(
            variable=args.variable,
            region=region,
            scenario=args.scenario,
            threshold=args.threshold
        )
    
    # Create processor using context manager for automatic cleanup
    with ModernCountyProcessor(
        n_workers=args.workers
    ) as processor:
        
        try:
            # Load shapefile
            console.print("[blue]📍 Loading county boundaries...[/blue]")
            gdf = processor.prepare_shapefile(args.shapefile_path)
            
            # Process data
            console.print(f"[blue]🔄 Processing {args.variable.upper()} data...[/blue]")
            results_df = processor.process_zarr_data(
                zarr_path=args.zarr_path,
                gdf=gdf,
                scenario=args.scenario,
                variable=args.variable,
                threshold=args.threshold
            )
            
            # Save results with metadata
            metadata = {
                "processing_info": {
                    "zarr_path": str(args.zarr_path),
                    "shapefile_path": str(args.shapefile_path),
                    "variable": args.variable,
                    "scenario": args.scenario,
                    "threshold": args.threshold,
                    "workers": args.workers
                },
                "data_summary": {
                    "counties_processed": len(results_df['county_id'].unique()),
                    "years_processed": len(results_df['year'].unique()),
                    "total_records": len(results_df)
                }
            }
            
            output_manager.save_with_metadata(
                data=results_df,
                output_path=args.output,
                metadata=metadata,
                save_method="csv"
            )
            
            # Show summary
            table = Table(title="📊 Processing Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Counties Processed", str(len(results_df['county_id'].unique())))
            table.add_row("Years Processed", str(len(results_df['year'].unique())))
            table.add_row("Total Records", str(len(results_df)))
            table.add_row("Variable", args.variable.upper())
            table.add_row("Output File", str(args.output))
            
            console.print(table)
            console.print("[green]✅ Processing completed successfully![/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error during processing: {e}[/red]")
            raise


if __name__ == "__main__":
    main() 