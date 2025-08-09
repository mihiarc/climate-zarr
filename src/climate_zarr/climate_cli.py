#!/usr/bin/env python
"""
🌡️ Climate Zarr CLI Tool - Modern 2025 Edition

A powerful CLI for processing climate data with NetCDF to Zarr conversion 
and county-level statistical analysis.

Features:
- Convert NetCDF files to optimized Zarr format
- Calculate detailed climate statistics by county/region
- Support for multiple climate variables (precipitation, temperature)
- Modern parallel processing with Rich progress bars
- Regional clipping with built-in boundary definitions
"""

import sys
from pathlib import Path
from typing import Optional, List
import warnings
import json
import matplotlib.pyplot as plt
from datetime import datetime

import typer
import questionary
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from typing_extensions import Annotated

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import our existing modules
from climate_zarr.stack_nc_to_zarr import stack_netcdf_to_zarr
from climate_zarr.county_processor import ModernCountyProcessor
from climate_zarr.utils.output_utils import get_output_manager
from climate_zarr.climate_config import get_config
from climate_zarr.visualization import ClimateVisualizer, discover_processed_data

# Initialize Rich console and Typer app
console = Console(highlight=False)
app = typer.Typer(
    name="climate-zarr",
    help="🌡️ Modern climate data processing toolkit",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Configuration
CONFIG = get_config()


def interactive_region_selection() -> str:
    """Interactive region selection with descriptions."""
    choices = []
    for region_key, region_config in CONFIG.regions.items():
        description = f"{region_config.name} ({region_config.lat_min:.1f}°N to {region_config.lat_max:.1f}°N)"
        choices.append(questionary.Choice(title=description, value=region_key))
    
    return questionary.select(
        "🗺️ Select a region:",
        choices=choices,
        style=questionary.Style([
            ('question', 'bold blue'),
            ('answer', 'bold green'),
            ('pointer', 'bold yellow'),
            ('highlighted', 'bold cyan'),
        ])
    ).ask()


def interactive_variable_selection() -> str:
    """Interactive climate variable selection."""
    variables = {
        'pr': '🌧️ Precipitation (mm/day) - rainfall and snowfall',
        'tas': '🌡️ Air Temperature (°C) - daily mean temperature',
        'tasmax': '🔥 Daily Maximum Temperature (°C) - highest daily temp',
        'tasmin': '🧊 Daily Minimum Temperature (°C) - lowest daily temp'
    }
    
    choices = [
        questionary.Choice(title=description, value=var)
        for var, description in variables.items()
    ]
    
    return questionary.select(
        "🔬 Select climate variable to analyze:",
        choices=choices,
        style=questionary.Style([
            ('question', 'bold blue'),
            ('answer', 'bold green'),
            ('pointer', 'bold yellow'),
            ('highlighted', 'bold cyan'),
        ])
    ).ask()


def interactive_file_selection() -> Path:
    """Interactive file/directory selection."""
    current_dir = Path.cwd()
    
    # Check for common data directories
    common_dirs = ['data', 'input', 'netcdf', 'nc_files']
    suggested_paths = []
    
    for dir_name in common_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists():
            nc_files = list(dir_path.glob("*.nc"))
            if nc_files:
                suggested_paths.append((dir_path, len(nc_files)))
    
    if suggested_paths:
        choices = []
        for path, count in suggested_paths:
            choices.append(
                questionary.Choice(
                    title=f"📁 {path.name}/ ({count} NetCDF files)", 
                    value=str(path)
                )
            )
        choices.append(questionary.Choice(title="📝 Enter custom path", value="custom"))
        
        selected = questionary.select(
            "📂 Select data source:",
            choices=choices,
            style=questionary.Style([
                ('question', 'bold blue'),
                ('answer', 'bold green'),
                ('pointer', 'bold yellow'),
                ('highlighted', 'bold cyan'),
            ])
        ).ask()
        
        if selected == "custom":
            return Path(questionary.path("Enter path to NetCDF files:").ask())
        else:
            return Path(selected)
    else:
        return Path(questionary.path("📂 Enter path to NetCDF files:").ask())


def discover_zarr_files() -> List[tuple]:
    """Discover existing Zarr files with metadata."""
    zarr_files = []
    current_dir = Path.cwd()
    
    # Common Zarr locations
    search_paths = [
        current_dir / "climate_outputs" / "zarr",
        current_dir / "zarr",
        current_dir,
    ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        # Find .zarr directories
        for zarr_path in search_path.rglob("*.zarr"):
            if zarr_path.is_dir():
                try:
                    # Try to open and get basic info
                    import xarray as xr
                    ds = xr.open_zarr(zarr_path, chunks={})
                    
                    # Extract metadata
                    variables = list(ds.data_vars.keys())
                    size_mb = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()) / (1024 * 1024)
                    
                    # Try to infer region from path
                    region = "unknown"
                    for region_name in ['conus', 'alaska', 'hawaii', 'guam', 'puerto_rico']:
                        if region_name in str(zarr_path).lower():
                            region = region_name
                            break
                    
                    zarr_files.append((
                        zarr_path,
                        variables,
                        region,
                        size_mb,
                        ds.dims.get('time', 0) if 'time' in ds.dims else 0
                    ))
                    ds.close()
                except Exception:
                    # If we can't read it, still show it but with limited info
                    zarr_files.append((zarr_path, ["unknown"], "unknown", 0, 0))
    
    return zarr_files


def interactive_zarr_selection() -> Path:
    """Interactive Zarr file selection with smart discovery."""
    console.print("[blue]🔍 Searching for existing Zarr files...[/blue]")
    
    zarr_files = discover_zarr_files()
    
    if zarr_files:
        choices = []
        for zarr_path, variables, region, size_mb, time_steps in zarr_files:
            # Create descriptive title
            var_str = ", ".join(variables[:3])  # Show first 3 variables
            if len(variables) > 3:
                var_str += f" (+{len(variables) - 3} more)"
            
            size_str = f"{size_mb:.1f}MB" if size_mb > 0 else "unknown size"
            time_str = f", {time_steps} time steps" if time_steps > 0 else ""
            
            title = f"📊 {zarr_path.name} ({var_str}, {region}, {size_str}{time_str})"
            choices.append(questionary.Choice(title=title, value=str(zarr_path)))
        
        choices.append(questionary.Choice(title="📝 Enter custom path", value="custom"))
        
        selected = questionary.select(
            "📂 Select Zarr dataset:",
            choices=choices,
            style=questionary.Style([
                ('question', 'bold blue'),
                ('answer', 'bold green'), 
                ('pointer', 'bold yellow'),
                ('highlighted', 'bold cyan'),
            ])
        ).ask()
        
        if selected == "custom":
            return Path(questionary.path("Enter path to Zarr dataset:").ask())
        else:
            return Path(selected)
    else:
        console.print("[yellow]ℹ️ No existing Zarr files found. Enter path manually.[/yellow]")
        return Path(questionary.path("📂 Enter path to Zarr dataset:").ask())


def validate_zarr_dataset(zarr_path: Path) -> tuple:
    """Validate Zarr dataset and return metadata."""
    try:
        import xarray as xr
        ds = xr.open_zarr(zarr_path, chunks={})
        
        variables = list(ds.data_vars.keys())
        dims = dict(ds.dims)
        coords = list(ds.coords.keys())
        
        # Check if it's a valid climate dataset
        has_time = 'time' in dims
        has_spatial = any(coord in coords for coord in ['lat', 'lon', 'x', 'y'])
        
        ds.close()
        return True, {
            'variables': variables,
            'dimensions': dims,
            'coordinates': coords,
            'has_time': has_time,
            'has_spatial': has_spatial
        }
    except Exception as e:
        return False, str(e)


def confirm_operation(operation: str, details: dict) -> bool:
    """Confirm potentially destructive operations."""
    console.print(f"\n[yellow]⚠️ About to {operation}[/yellow]")
    
    # Show operation details
    details_table = Table(show_header=False, border_style="yellow")
    details_table.add_column("Setting", style="cyan")
    details_table.add_column("Value", style="white")
    
    for key, value in details.items():
        details_table.add_row(key, str(value))
    
    console.print(details_table)
    
    return questionary.confirm(
        f"🤔 Proceed with {operation}?",
        default=False,
        style=questionary.Style([
            ('question', 'bold yellow'),
            ('answer', 'bold green'),
        ])
    ).ask()


def print_banner():
    """Display a beautiful banner."""
    banner = Panel.fit(
        "[bold blue]🌡️ Climate Zarr Toolkit[/bold blue]\n"
        "[dim]Modern NetCDF → Zarr conversion & county statistics[/dim]",
        border_style="blue",
    )
    console.print(banner)


def validate_region(region: str) -> str:
    """Validate region name against available regions."""
    if region is None:
        return region
    
    available_regions = list(CONFIG.regions.keys())
    if region.lower() not in available_regions:
        rprint(f"[red]❌ Unknown region: {region}[/red]")
        rprint(f"[yellow]Available regions:[/yellow] {', '.join(available_regions)}")
        
        # Interactive suggestion
        if questionary.confirm("🤔 Would you like to select from available regions?").ask():
            return interactive_region_selection()
        else:
            raise typer.Exit(1)
    return region.lower()


def get_shapefile_for_region(region: str) -> Path:
    """Get the appropriate shapefile path for a region."""
    region_files = {
        'conus': 'conus_counties.shp',
        'alaska': 'alaska_counties.shp', 
        'hawaii': 'hawaii_counties.shp',
        'guam': 'guam_counties.shp',
        'puerto_rico': 'puerto_rico_counties.shp',
        'pr_vi': 'puerto_rico_counties.shp',
        'other': 'other_counties.shp'
    }
    
    shapefile_name = region_files.get(region, f'{region}_counties.shp')
    shapefile_path = Path('regional_counties') / shapefile_name
    
    if not shapefile_path.exists():
        rprint(f"[red]❌ Shapefile not found: {shapefile_path}[/red]")
        raise typer.Exit(1)
    
    return shapefile_path


@app.command("create-zarr")
def create_zarr(
    input_path: Annotated[Optional[Path], typer.Argument(help="Directory containing NetCDF files or single NetCDF file")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output Zarr store path")] = None,
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="Clip data to specific region")] = None,
    concat_dim: Annotated[str, typer.Option("--concat-dim", "-d", help="Dimension to concatenate along")] = "time",
    chunks: Annotated[Optional[str], typer.Option("--chunks", "-c", help="Chunk sizes as 'dim1=size1,dim2=size2'")] = None,
    compression: Annotated[str, typer.Option("--compression", help="Compression algorithm")] = "default",
    compression_level: Annotated[int, typer.Option("--compression-level", help="Compression level (1-9)")] = 5,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Use interactive prompts for missing options")] = True,
):
    """
    🗜️ Convert NetCDF files to optimized Zarr format.
    
    This command stacks multiple NetCDF files into a single, compressed Zarr store
    with optimal chunking for analysis workflows.
    
    Examples:
        climate-zarr create-zarr  # Interactive mode
        climate-zarr create-zarr data/ -o precipitation.zarr --region conus
        climate-zarr create-zarr data/ -o temp.zarr --chunks "time=365,lat=180,lon=360"
    """
    print_banner()
    
    # Interactive prompts for missing parameters
    if not input_path and interactive:
        input_path = interactive_file_selection()
    elif not input_path:
        rprint("[red]❌ Input path is required[/red]")
        raise typer.Exit(1)
    
    if not output and interactive:
        suggested = Path(f"{input_path.stem}_climate.zarr" if input_path.is_file() else "climate_data.zarr")
        output = Path(Prompt.ask("📁 Output Zarr file", default=str(suggested)))
    elif not output:
        output = Path("climate_data.zarr")
    
    if not region and interactive:
        if Confirm.ask("🗺️ Clip data to a specific region?"):
            region = interactive_region_selection()
    
    # Validate region if specified
    if region:
        region = validate_region(region)
    
    # Collect NetCDF files
    nc_files = []
    if input_path.is_dir():
        nc_files = list(input_path.glob("*.nc"))
    elif input_path.is_file() and input_path.suffix == '.nc':
        nc_files = [input_path]
    else:
        rprint(f"[red]❌ No NetCDF files found in: {input_path}[/red]")
        raise typer.Exit(1)
    
    if not nc_files:
        rprint(f"[red]❌ No .nc files found in directory: {input_path}[/red]")
        raise typer.Exit(1)
    
    # Parse chunks if provided
    chunks_dict = None
    if chunks:
        chunks_dict = {}
        for chunk in chunks.split(','):
            key, value = chunk.split('=')
            chunks_dict[key.strip()] = int(value.strip())
    
    # Confirmation for large datasets
    if len(nc_files) > 50 and interactive:
        if not Confirm.ask(f"⚠️ Process {len(nc_files)} files? This may take a while."):
            console.print("[yellow]❌ Operation cancelled[/yellow]")
            raise typer.Exit(0)
    
    # Display processing info
    info_table = Table(title="📊 Processing Configuration", show_header=False)
    info_table.add_column("Setting", style="cyan")
    info_table.add_column("Value", style="green")
    
    info_table.add_row("Input Files", f"{len(nc_files)} NetCDF files")
    info_table.add_row("Output", str(output))
    info_table.add_row("Region", region if region else "Global (no clipping)")
    info_table.add_row("Concat Dimension", concat_dim)
    info_table.add_row("Compression", f"{compression} (level {compression_level})")
    if chunks_dict:
        chunks_str = ", ".join(f"{k}={v}" for k, v in chunks_dict.items())
        info_table.add_row("Chunks", chunks_str)
    
    console.print(info_table)
    console.print()
    
    try:
        # Run the conversion
        stack_netcdf_to_zarr(
            nc_files=nc_files,
            zarr_path=output,
            concat_dim=concat_dim,
            chunks=chunks_dict,
            compression=compression,
            compression_level=compression_level,
            clip_region=region
        )
        
        # Success message
        success_panel = Panel(
            f"[green]✅ Successfully created Zarr store: {output}[/green]",
            border_style="green"
        )
        console.print(success_panel)
        
    except Exception as e:
        rprint(f"[red]❌ Error creating Zarr store: {e}[/red]")
        raise typer.Exit(1)


@app.command("county-stats")
def county_stats(
    zarr_path: Annotated[Optional[Path], typer.Argument(help="Path to Zarr dataset")] = None,
    region: Annotated[Optional[str], typer.Argument(help="Region name (conus, alaska, hawaii, etc.)")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output CSV file")] = None,
    variable: Annotated[Optional[str], typer.Option("--variable", "-v", help="Climate variable to analyze")] = None,
    scenario: Annotated[str, typer.Option("--scenario", "-s", help="Scenario name")] = "historical",
    threshold: Annotated[Optional[float], typer.Option("--threshold", "-t", help="Threshold value")] = None,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of worker processes")] = 4,
    use_distributed: Annotated[bool, typer.Option("--distributed", help="Use Dask distributed processing")] = False,
    chunk_by_county: Annotated[bool, typer.Option("--chunk-counties", help="Process counties in chunks")] = True,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Use interactive prompts for missing options")] = True,
):
    """
    📈 Calculate detailed climate statistics by county for a specific region.
    
    Analyzes climate data and generates comprehensive statistics for each county
    in the specified region, with support for multiple climate variables.
    
    Examples:
        climate-zarr county-stats  # Interactive mode
        climate-zarr county-stats precipitation.zarr conus -v pr -t 25.4
        climate-zarr county-stats temperature.zarr alaska -v tas --workers 8
    """
    print_banner()
    
    # Interactive prompts for missing parameters
    if not zarr_path and interactive:
        zarr_path = Path(Prompt.ask("📁 Path to Zarr dataset"))
    elif not zarr_path:
        rprint("[red]❌ Zarr path is required[/red]")
        raise typer.Exit(1)
    
    if not zarr_path.exists():
        rprint(f"[red]❌ Zarr dataset not found: {zarr_path}[/red]")
        raise typer.Exit(1)
    
    if not region and interactive:
        region = interactive_region_selection()
    elif not region:
        rprint("[red]❌ Region is required[/red]")
        raise typer.Exit(1)
    
    region = validate_region(region)
    
    if not variable and interactive:
        variable = interactive_variable_selection()
    elif not variable:
        variable = "pr"
    shapefile_path = get_shapefile_for_region(region)
    
    # Variable validation
    valid_variables = ["pr", "tas", "tasmax", "tasmin"]
    if variable not in valid_variables:
        rprint(f"[red]❌ Invalid variable: {variable}[/red]")
        rprint(f"[yellow]Valid variables:[/yellow] {', '.join(valid_variables)}")
        raise typer.Exit(1)
    
    if threshold is None and interactive:
        default_threshold = "25.4" if variable == "pr" else "32" if variable == "tasmax" else "0"
        threshold_str = Prompt.ask(
            f"🎯 Threshold value ({'mm/day' if variable == 'pr' else '°C'})",
            default=default_threshold
        )
        threshold = float(threshold_str)
    elif threshold is None:
        threshold = 25.4 if variable == "pr" else 32.0 if variable == "tasmax" else 0.0
    
    if not output and interactive:
        # Use output manager to suggest standardized filename
        output_manager = get_output_manager()
        suggested_path = output_manager.get_output_path(
            variable=variable,
            region=region,
            scenario=scenario,
            threshold=threshold
        )
        output = Path(Prompt.ask(
            "📊 Output CSV file",
            default=str(suggested_path)
        ))
    elif not output:
        # Auto-generate standardized output path
        output_manager = get_output_manager()
        output = output_manager.get_output_path(
            variable=variable,
            region=region,
            scenario=scenario,
            threshold=threshold
        )
    
    # Confirmation for large operations
    if interactive and workers > 8:
        if not Confirm.ask(f"⚠️ Use {workers} workers? This will use significant system resources."):
            workers = 4
            console.print("[yellow]🔧 Reduced to 4 workers[/yellow]")
    
    # Display processing configuration
    config_table = Table(title="🔧 Analysis Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Zarr Dataset", str(zarr_path))
    config_table.add_row("Region", region.upper())
    config_table.add_row("Shapefile", str(shapefile_path))
    config_table.add_row("Variable", variable.upper())
    config_table.add_row("Scenario", scenario)
    config_table.add_row("Threshold", f"{threshold} {'mm' if variable == 'pr' else '°C'}")
    config_table.add_row("Workers", str(workers))
    config_table.add_row("Processing", "Distributed" if use_distributed else "Multiprocessing")
    config_table.add_row("Output", str(output))
    
    console.print(config_table)
    console.print()
    
    try:
        # Create processor
        processor = ModernCountyProcessor(
            n_workers=workers
        )
        
        # Load shapefile
        console.print("[blue]📍 Loading county boundaries...[/blue]")
        gdf = processor.prepare_shapefile(shapefile_path)
        
        # Process data
        console.print(f"[blue]🔄 Processing {variable.upper()} data for {len(gdf)} counties...[/blue]")
        results_df = processor.process_zarr_data(
            zarr_path=zarr_path,
            gdf=gdf,
            scenario=scenario,
            variable=variable,
            threshold=threshold,
            chunk_by_county=chunk_by_county
        )
        
        # Save results with metadata
        if 'output_manager' not in locals():
            output_manager = get_output_manager()
        
        metadata = {
            "processing_info": {
                "zarr_path": str(zarr_path),
                "shapefile_path": str(shapefile_path),
                "variable": variable,
                "scenario": scenario,
                "threshold": threshold,
                "workers": workers,
                "use_distributed": use_distributed,
                "chunk_by_county": chunk_by_county
            },
            "data_summary": {
                "counties_processed": len(results_df['county_id'].unique()),
                "years_analyzed": len(results_df['year'].unique()),
                "total_records": len(results_df)
            }
        }
        
        output_manager.save_with_metadata(
            data=results_df,
            output_path=output,
            metadata=metadata,
            save_method="csv"
        )
        
        # Display summary
        summary_table = Table(title="📊 Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        summary_table.add_row("Counties Processed", str(len(results_df['county_id'].unique())))
        summary_table.add_row("Years Analyzed", str(len(results_df['year'].unique())))
        summary_table.add_row("Total Records", str(len(results_df)))
        summary_table.add_row("Variable", variable.upper())
        summary_table.add_row("Output File", str(output))
        
        console.print(summary_table)
        
        # Success message
        success_panel = Panel(
            f"[green]✅ County statistics saved to: {output}[/green]",
            border_style="green"
        )
        console.print(success_panel)
        
        # Clean up
        processor.close()
        
    except Exception as e:
        rprint(f"[red]❌ Error processing county statistics: {e}[/red]")
        raise typer.Exit(1)


@app.command("wizard")
def interactive_wizard():
    """
    🧙‍♂️ Launch the interactive wizard for guided climate data processing.
    
    This wizard will guide you through the entire process step-by-step,
    from selecting data to generating results.
    """
    print_banner()
    
    console.print(Panel(
        "[bold cyan]🧙‍♂️ Welcome to the Climate Data Processing Wizard![/bold cyan]\n\n"
        "This interactive guide will help you:\n"
        "• Convert NetCDF files to optimized Zarr format\n"
        "• Calculate detailed county statistics\n"
        "• Choose the best settings for your analysis\n\n"
        "[dim]Let's get started![/dim]",
        border_style="cyan"
    ))
    
    # Step 1: Choose operation
    operation = questionary.select(
        "🎯 What would you like to do?",
        choices=[
            questionary.Choice("🗜️ Convert NetCDF to Zarr", "convert"),
            questionary.Choice("📈 Calculate county statistics", "stats"),
            questionary.Choice("🔄 Full pipeline (convert + analyze)", "pipeline"),
            questionary.Choice("ℹ️ Just show me information", "info"),
        ],
        style=questionary.Style([
            ('question', 'bold blue'),
            ('answer', 'bold green'),
            ('pointer', 'bold yellow'),
            ('highlighted', 'bold cyan'),
        ])
    ).ask()
    
    if operation == "info":
        info()
        return
    elif operation == "stats":
        # Skip NetCDF selection for stats-only workflow
        console.print("\n[bold blue]📊 Step 1: Select Zarr dataset[/bold blue]")
        zarr_path = interactive_zarr_selection()
        
        if not zarr_path.exists():
            console.print(f"[red]❌ Zarr dataset not found: {zarr_path}[/red]")
            return
        
        # Validate Zarr dataset
        console.print("[blue]🔍 Validating Zarr dataset...[/blue]")
        is_valid, metadata = validate_zarr_dataset(zarr_path)
        
        if not is_valid:
            console.print(f"[red]❌ Invalid Zarr dataset: {metadata}[/red]")
            return
        
        # Show dataset info
        info_table = Table(title="📊 Zarr Dataset Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Variables", ", ".join(metadata['variables']))
        info_table.add_row("Dimensions", ", ".join([f"{k}={v}" for k, v in metadata['dimensions'].items()]))
        info_table.add_row("Has Time Dimension", "✅ Yes" if metadata['has_time'] else "❌ No")
        info_table.add_row("Has Spatial Coords", "✅ Yes" if metadata['has_spatial'] else "❌ No")
        
        console.print(info_table)
        
        # Set variables for later use
        nc_files = []  # Not needed for stats-only
        input_path = zarr_path.parent  # For context
        
    else:
        # NetCDF selection for convert/pipeline operations
        console.print("\n[bold blue]📁 Step 1: Select your data[/bold blue]")
        input_path = interactive_file_selection()
        
        if not input_path.exists():
            console.print(f"[red]❌ Path does not exist: {input_path}[/red]")
            return
        
        # Collect NetCDF files
        nc_files = []
        if input_path.is_dir():
            nc_files = list(input_path.glob("*.nc"))
        elif input_path.is_file() and input_path.suffix == '.nc':
            nc_files = [input_path]
        
        if not nc_files:
            console.print(f"[red]❌ No NetCDF files found in: {input_path}[/red]")
            return
        
        console.print(f"[green]✅ Found {len(nc_files)} NetCDF files[/green]")
    
    if operation in ["convert", "pipeline"]:
        # Step 3: Conversion settings
        console.print("\n[bold blue]🗜️ Step 2: Configure Zarr conversion[/bold blue]")
        
        # Output path
        suggested_output = Path(f"{input_path.stem}_climate.zarr" if input_path.is_file() else "climate_data.zarr")
        output_path = Path(questionary.text(
            "📁 Output Zarr file name:",
            default=str(suggested_output)
        ).ask())
        
        # Region selection
        use_region = questionary.confirm("🗺️ Clip data to a specific region?", default=True).ask()
        region = None
        if use_region:
            region = interactive_region_selection()
        
        # Compression
        compression = questionary.select(
            "🗜️ Choose compression algorithm:",
            choices=[
                questionary.Choice("🚀 ZSTD (recommended - fast & efficient)", "zstd"),
                questionary.Choice("📦 Default (Blosc)", "default"), 
                questionary.Choice("🔧 ZLIB (compatible)", "zlib"),
                questionary.Choice("📄 GZIP (universal)", "gzip"),
            ]
        ).ask()
        
        # Confirm conversion
        conversion_details = {
            "Input Files": f"{len(nc_files)} NetCDF files",
            "Output": str(output_path),
            "Region": region.upper() if region else "Global (no clipping)",
            "Compression": compression,
        }
        
        if not confirm_operation("convert NetCDF to Zarr", conversion_details):
            console.print("[yellow]❌ Operation cancelled by user[/yellow]")
            return
        
        # Perform conversion
        try:
            console.print("\n[blue]🔄 Converting NetCDF files to Zarr...[/blue]")
            
            stack_netcdf_to_zarr(
                nc_files=nc_files,
                zarr_path=output_path,
                concat_dim="time",
                chunks=None,
                compression=compression.split()[0],  # Extract algorithm name
                compression_level=5,
                clip_region=region
            )
            
            console.print(Panel(
                f"[green]✅ Successfully created Zarr store: {output_path}[/green]",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]❌ Error during conversion: {e}[/red]")
            return
    
    if operation in ["stats", "pipeline"]:
        # Step 4: Statistics configuration  
        step_num = "2" if operation == "stats" else "3"
        console.print(f"\n[bold blue]📈 Step {step_num}: Configure county statistics[/bold blue]")
        
        # Use existing zarr path (already selected for stats, created for pipeline)
        if operation == "pipeline":
            zarr_path = output_path
        
        # Region for statistics
        stats_region = interactive_region_selection()
        
        # Smart variable selection based on available data
        if operation == "stats" and 'variables' in locals():
            # Filter available variables from the dataset
            available_vars = [v for v in metadata['variables'] if v in ['pr', 'tas', 'tasmax', 'tasmin']]
            if available_vars:
                if len(available_vars) == 1:
                    variable = available_vars[0]
                    console.print(f"[green]📊 Using detected variable: {variable.upper()}[/green]")
                else:
                    variable = questionary.select(
                        "🔬 Select variable from your dataset:",
                        choices=[
                            questionary.Choice(
                                title=f"{'🌧️' if v == 'pr' else '🌡️'} {v.upper()}", 
                                value=v
                            ) for v in available_vars
                        ]
                    ).ask()
            else:
                console.print("[yellow]⚠️ No standard climate variables found in dataset. Showing all options.[/yellow]")
                variable = interactive_variable_selection()
        else:
            variable = interactive_variable_selection()
        
        # Threshold configuration
        if variable == "pr":
            threshold = questionary.text(
                "🌧️ Precipitation threshold (mm/day):",
                default="25.4",
                validate=lambda x: x.replace('.', '').isdigit()
            ).ask()
        elif variable in ["tasmax", "tasmin"]:
            threshold = questionary.text(
                f"🌡️ Temperature threshold (°C):",
                default="32" if variable == "tasmax" else "0",
                validate=lambda x: x.replace('.', '').replace('-', '').isdigit()
            ).ask()
        else:
            threshold = "0"
        
        # Output file using standardized naming
        output_manager = get_output_manager()
        suggested_output = output_manager.get_output_path(
            variable=variable,
            region=stats_region,
            scenario="historical",
            threshold=float(threshold)
        )
        output_csv = Path(questionary.text(
            "📊 Output CSV file name:",
            default=str(suggested_output)
        ).ask())
        
        # Performance settings
        workers = questionary.select(
            "⚡ Number of worker processes:",
            choices=["2", "4", "8", "16"],
            default="4"
        ).ask()
        
        use_distributed = questionary.confirm(
            "🚀 Use distributed processing? (for very large datasets)",
            default=False
        ).ask()
        
        # Confirm statistics calculation
        stats_details = {
            "Zarr Dataset": str(zarr_path),
            "Region": stats_region.upper(),
            "Variable": variable.upper(),
            "Threshold": f"{threshold} {'mm' if variable == 'pr' else '°C'}",
            "Workers": workers,
            "Processing": "Distributed" if use_distributed else "Multiprocessing",
            "Output": str(output_csv),
        }
        
        if not confirm_operation("calculate county statistics", stats_details):
            console.print("[yellow]❌ Operation cancelled by user[/yellow]")
            return
        
        # Perform statistics calculation
        try:
            # Get shapefile path
            shapefile_path = get_shapefile_for_region(stats_region)
            
            # Create processor
            processor = ModernCountyProcessor(
                n_workers=int(workers)
            )
            
            # Load shapefile
            console.print("[blue]📍 Loading county boundaries...[/blue]")
            gdf = processor.prepare_shapefile(shapefile_path)
            
            # Process data
            console.print(f"[blue]🔄 Processing {variable.upper()} data for {len(gdf)} counties...[/blue]")
            results_df = processor.process_zarr_data(
                zarr_path=zarr_path,
                gdf=gdf,
                scenario="historical",
                variable=variable,
                threshold=float(threshold),
                chunk_by_county=True
            )
            
            # Save results with metadata
            metadata = {
                "processing_info": {
                    "zarr_path": str(zarr_path),
                    "shapefile_path": str(shapefile_path),
                    "variable": variable,
                    "scenario": "historical",
                    "threshold": float(threshold),
                    "workers": int(workers),
                    "use_distributed": use_distributed
                },
                "data_summary": {
                    "counties_processed": len(results_df['county_id'].unique()),
                    "years_analyzed": len(results_df['year'].unique()),
                    "total_records": len(results_df)
                }
            }
            
            output_manager.save_with_metadata(
                data=results_df,
                output_path=output_csv,
                metadata=metadata,
                save_method="csv"
            )
            
            # Show success summary
            summary_table = Table(title="📊 Processing Complete!")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="magenta")
            
            summary_table.add_row("Counties Processed", str(len(results_df['county_id'].unique())))
            summary_table.add_row("Years Analyzed", str(len(results_df['year'].unique())))
            summary_table.add_row("Total Records", str(len(results_df)))
            summary_table.add_row("Output File", str(output_csv))
            
            console.print(summary_table)
            
            # Success message
            console.print(Panel(
                f"[green]✅ County statistics saved to: {output_csv}[/green]",
                border_style="green"
            ))
            
            # Clean up
            processor.close()
            
        except Exception as e:
            console.print(f"[red]❌ Error processing statistics: {e}[/red]")
            return
    
    # Final success message
    console.print("\n" + "🎉" * 50)
    console.print(Panel(
        "[bold green]🎊 Wizard completed successfully![/bold green]\n\n"
        "[cyan]What you accomplished:[/cyan]\n"
        f"• {'✅ Converted NetCDF to Zarr format' if operation in ['convert', 'pipeline'] else ''}\n"
        f"• {'✅ Calculated detailed county statistics' if operation in ['stats', 'pipeline'] else ''}\n"
        f"• {'✅ Processed ' + str(len(nc_files)) + ' NetCDF files' if nc_files else ''}\n\n"
        "[dim]🚀 You're ready to explore your climate data![/dim]",
        border_style="green",
        title="🏆 Success"
    ))


@app.command("interactive")  
def interactive_mode():
    """
    🎮 Enter interactive mode for guided climate data processing.
    
    This launches an interactive session where you can explore data,
    run commands, and get guided assistance.
    """
    interactive_wizard()


@app.command("list-regions")
def list_regions():
    """📍 List all available regions for clipping and analysis."""
    print_banner()
    
    regions_table = Table(title="🗺️ Available Regions")
    regions_table.add_column("Region", style="cyan")
    regions_table.add_column("Name", style="green")
    regions_table.add_column("Boundaries (Lat/Lon)", style="yellow")
    
    for region_key, region_config in CONFIG.regions.items():
        bounds = f"{region_config.lat_min:.1f}°N to {region_config.lat_max:.1f}°N, "
        bounds += f"{region_config.lon_min:.1f}°E to {region_config.lon_max:.1f}°E"
        
        regions_table.add_row(
            region_key,
            region_config.name,
            bounds
        )
    
    console.print(regions_table)


@app.command("info")
def info():
    """ℹ️ Display system information and available data."""
    print_banner()
    
    # Check data directory
    data_dir = Path("data")
    nc_files = list(data_dir.glob("*.nc")) if data_dir.exists() else []
    
    # Check regional counties
    regional_dir = Path("regional_counties")
    shapefiles = list(regional_dir.glob("*.shp")) if regional_dir.exists() else []
    
    # System info
    info_layout = Layout()
    info_layout.split_column(
        Layout(name="data"),
        Layout(name="regions")
    )
    
    # Data info
    data_table = Table(title="📁 Available Data")
    data_table.add_column("Type", style="cyan")
    data_table.add_column("Count", style="green")
    data_table.add_column("Location", style="yellow")
    
    data_table.add_row("NetCDF Files", str(len(nc_files)), str(data_dir))
    data_table.add_row("Regional Shapefiles", str(len(shapefiles)), str(regional_dir))
    
    # Regions info
    regions_table = Table(title="🗺️ Configured Regions")
    regions_table.add_column("Region", style="cyan")
    regions_table.add_column("Coverage", style="green")
    
    for region_key, region_config in CONFIG.regions.items():
        regions_table.add_row(region_key, region_config.name)
    
    info_layout["data"].update(Panel(data_table, border_style="blue"))
    info_layout["regions"].update(Panel(regions_table, border_style="green"))
    
    console.print(info_layout)
    
    # Sample NetCDF files
    if nc_files:
        console.print(f"\n[dim]Sample NetCDF files (showing first 5):[/dim]")
        for nc_file in nc_files[:5]:
            console.print(f"  • {nc_file.name}")
        if len(nc_files) > 5:
            console.print(f"  ... and {len(nc_files) - 5} more")


@app.command("visualize")
def create_visualizations(
    data_dir: Annotated[Optional[Path], typer.Option("--data-dir", "-d", help="Directory containing processed climate data")] = None,
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir", "-o", help="Directory for saving plots and reports")] = None,
    regions: Annotated[Optional[str], typer.Option("--regions", "-r", help="Comma-separated list of regions to analyze")] = None,
    plot_types: Annotated[str, typer.Option("--plots", "-p", help="Types of plots to generate")] = "all",
    interactive: Annotated[bool, typer.Option("--interactive/--no-interactive", "-i", help="Use interactive mode for options")] = True,
):
    """
    📊 Generate comprehensive visualizations and reports for processed climate data.
    
    Creates temporal trend plots, regional comparisons, climate change signals,
    and detailed summary reports from your processed county statistics.
    
    Examples:
        climate-zarr visualize  # Interactive mode
        climate-zarr visualize --data-dir climate_outputs --regions conus,alaska
        climate-zarr visualize --plots trends,comparison --output-dir reports
    """
    print_banner()
    
    # Set default data directory
    if not data_dir and interactive:
        # Try to auto-discover
        suggested_dir = Path("./climate_outputs")
        if suggested_dir.exists():
            data_dir = Path(Prompt.ask("📁 Data directory", default=str(suggested_dir)))
        else:
            data_dir = Path(Prompt.ask("📁 Directory containing processed climate data"))
    elif not data_dir:
        data_dir = Path("./climate_outputs")
    
    # Set output directory
    if not output_dir and interactive:
        suggested_output = Path("./climate_reports")
        output_dir = Path(Prompt.ask("📈 Output directory for reports", default=str(suggested_output)))
    elif not output_dir:
        output_dir = Path("./climate_reports")
    
    # Discover processed data
    console.print(f"[blue]🔍 Searching for processed data in: {data_dir}[/blue]")
    data_paths = discover_processed_data(data_dir)
    
    if not data_paths:
        console.print(f"[red]❌ No processed climate data found in: {data_dir}[/red]")
        console.print("[yellow]💡 Run 'climate-zarr county-stats' first to process data[/yellow]")
        raise typer.Exit(1)
    
    # Filter regions if specified
    if regions:
        selected_regions = [r.strip().lower() for r in regions.split(',')]
        data_paths = {r: p for r, p in data_paths.items() if r in selected_regions}
    elif interactive and len(data_paths) > 1:
        # Interactive region selection
        available_regions = list(data_paths.keys())
        if questionary.confirm("🗺️ Select specific regions to analyze?", default=False).ask():
            selected = questionary.checkbox(
                "Select regions:",
                choices=[questionary.Choice(r.upper(), r) for r in available_regions],
                style=questionary.Style([
                    ('question', 'bold blue'),
                    ('answer', 'bold green'),
                    ('pointer', 'bold yellow'),
                    ('highlighted', 'bold cyan'),
                ])
            ).ask()
            if selected:
                data_paths = {r: data_paths[r] for r in selected}
    
    console.print(f"[green]✅ Found data for {len(data_paths)} regions:[/green]")
    data_table = Table(show_header=True, header_style="bold blue")
    data_table.add_column("Region", style="cyan")
    data_table.add_column("Data File", style="green")
    data_table.add_column("File Size", style="yellow")
    
    for region, path in data_paths.items():
        file_size = f"{path.stat().st_size / (1024*1024):.1f} MB" if path.exists() else "N/A"
        data_table.add_row(region.upper(), path.name, file_size)
    
    console.print(data_table)
    
    # Plot type selection
    available_plots = {
        'trends': 'Temporal trends analysis',
        'comparison': 'Regional comparison plots', 
        'signals': 'Climate change signals',
        'all': 'All visualization types'
    }
    
    if plot_types == "all" and interactive:
        selected_plots = questionary.checkbox(
            "📊 Select visualization types:",
            choices=[questionary.Choice(f"{desc}", value) for value, desc in available_plots.items() if value != 'all'],
            default=['trends', 'comparison', 'signals'],
            style=questionary.Style([
                ('question', 'bold blue'),
                ('answer', 'bold green'),
                ('pointer', 'bold yellow'),
                ('highlighted', 'bold cyan'),
            ])
        ).ask()
        plot_types = ','.join(selected_plots) if selected_plots else 'all'
    
    # Confirmation
    if interactive:
        viz_details = {
            "Data Directory": str(data_dir),
            "Output Directory": str(output_dir),
            "Regions": ", ".join(data_paths.keys()).upper(),
            "Plot Types": plot_types.replace(',', ', ').title(),
            "Total Records": "~214k+" if len(data_paths) > 1 else "varies"
        }
        
        if not confirm_operation("generate visualizations and reports", viz_details):
            console.print("[yellow]❌ Operation cancelled by user[/yellow]")
            raise typer.Exit(0)
    
    try:
        # Create visualizer
        console.print(f"[blue]🎨 Initializing visualization system...[/blue]")
        visualizer = ClimateVisualizer(output_dir)
        
        # Load data
        console.print(f"[blue]📊 Loading regional climate data...[/blue]")
        loaded_data = visualizer.load_regional_data(data_paths)
        
        if not loaded_data:
            console.print("[red]❌ Failed to load any data[/red]")
            raise typer.Exit(1)
        
        # Generate visualizations based on selection
        plot_list = plot_types.split(',') if plot_types != 'all' else ['trends', 'comparison', 'signals']
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            viz_task = progress.add_task("Creating visualizations...", total=len(plot_list) + 1)
            
            generated_plots = []
            
            if 'trends' in plot_list or plot_types == 'all':
                progress.update(viz_task, description="Creating temporal trends plots...")
                fig = visualizer.create_temporal_trends_plot()
                generated_plots.append("temporal_trends")
                plt.close(fig)  # Free memory
                progress.advance(viz_task)
            
            if 'comparison' in plot_list or plot_types == 'all':
                progress.update(viz_task, description="Creating regional comparison plots...")
                fig = visualizer.create_regional_comparison_plot()
                generated_plots.append("regional_comparison")
                plt.close(fig)  # Free memory
                progress.advance(viz_task)
            
            if 'signals' in plot_list or plot_types == 'all':
                progress.update(viz_task, description="Creating climate change signals plots...")
                fig = visualizer.create_climate_change_signals_plot()
                generated_plots.append("climate_change_signals")
                plt.close(fig)  # Free memory
                progress.advance(viz_task)
            
            # Generate comprehensive report
            progress.update(viz_task, description="Generating comprehensive report...")
            report = visualizer.generate_summary_report()
            progress.advance(viz_task)
        
        # Display results summary
        results_table = Table(title="📊 Visualization Results")
        results_table.add_column("Output Type", style="cyan")
        results_table.add_column("Files Generated", style="green")
        results_table.add_column("Location", style="yellow")
        
        plot_count = len(list((output_dir / 'plots').glob('*.png')))
        report_count = len(list((output_dir / 'reports').glob('*')))
        
        results_table.add_row("Plots", str(plot_count), str(output_dir / 'plots'))
        results_table.add_row("Reports", str(report_count), str(output_dir / 'reports'))
        results_table.add_row("Total Records Analyzed", f"{sum(len(df) for df in loaded_data.values()):,}", "")
        
        console.print(results_table)
        
        # Success message
        success_panel = Panel(
            f"[green]✅ Successfully generated climate visualizations and reports![/green]\n\n"
            f"[cyan]Generated:[/cyan]\n"
            f"• {plot_count} visualization plots\n"
            f"• {report_count} comprehensive reports\n"
            f"• Analysis of {sum(len(df) for df in loaded_data.values()):,} climate records\n\n"
            f"[yellow]📁 All outputs saved to: {output_dir}[/yellow]",
            border_style="green",
            title="🎊 Analysis Complete"
        )
        console.print(success_panel)
        
    except Exception as e:
        console.print(f"[red]❌ Error during visualization: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command("report")
def generate_report(
    data_dir: Annotated[Optional[Path], typer.Option("--data-dir", "-d", help="Directory containing processed climate data")] = None,
    output_file: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output report file")] = None,
    format: Annotated[str, typer.Option("--format", "-f", help="Report format")] = "json",
    regions: Annotated[Optional[str], typer.Option("--regions", "-r", help="Comma-separated list of regions")] = None,
    interactive: Annotated[bool, typer.Option("--interactive/--no-interactive", "-i", help="Use interactive mode")] = True,
):
    """
    📄 Generate detailed summary reports for processed climate data.
    
    Creates comprehensive analysis reports with statistics, trends, and 
    comparative analysis across regions and time periods.
    
    Examples:
        climate-zarr report  # Interactive mode
        climate-zarr report --format json --regions conus,alaska
        climate-zarr report --output climate_analysis.json
    """
    print_banner()
    
    if not data_dir:
        data_dir = Path("./climate_outputs")
    
    # Discover and analyze data
    data_paths = discover_processed_data(data_dir)
    
    if not data_paths:
        console.print(f"[red]❌ No processed data found in: {data_dir}[/red]")
        raise typer.Exit(1)
    
    # Filter regions if specified
    if regions:
        selected_regions = [r.strip().lower() for r in regions.split(',')]
        data_paths = {r: p for r, p in data_paths.items() if r in selected_regions}
    
    # Create temporary visualizer for report generation
    temp_output_dir = Path("./temp_reports")
    visualizer = ClimateVisualizer(temp_output_dir)
    
    # Load data and generate report
    visualizer.load_regional_data(data_paths)
    report = visualizer.generate_summary_report(save_report=False)
    
    # Save report in requested format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_file:
        output_file = Path(f"climate_report_{timestamp}.{format}")
    
    if format.lower() == "json":
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    elif format.lower() == "txt":
        visualizer._create_readable_summary(report, timestamp)
        # Move the file to desired location
        temp_file = temp_output_dir / 'reports' / f'climate_summary_readable_{timestamp}.txt'
        if temp_file.exists():
            temp_file.rename(output_file)
    
    console.print(f"[green]✅ Report saved to: {output_file}[/green]")
    
    # Cleanup
    import shutil
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)


if __name__ == "__main__":
    # Prefer modular CLI if available
    try:
        from climate_zarr.cli import app as _modular_app
        _modular_app()
    except Exception:
        app()