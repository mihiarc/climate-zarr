#!/usr/bin/env python
"""Simplified CLI for county statistics using the modular processor."""

import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

from climate_zarr.county_processor import ModernCountyProcessor

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
        default="county_stats.csv",
        help="Output CSV file (default: county_stats.csv)"
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

    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Disable chunked processing"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.zarr_path.exists():
        console.print(f"[red]Zarr path does not exist: {args.zarr_path}[/red]")
        return
    
    if not args.shapefile_path.exists():
        console.print(f"[red]Shapefile does not exist: {args.shapefile_path}[/red]")
        return
    
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
                threshold=args.threshold,
                chunk_by_county=not args.no_chunk
            )
            
            # Save results
            results_df.to_csv(args.output, index=False)
            
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