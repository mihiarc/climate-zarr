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
from climate_zarr.calculate_county_stats import ModernCountyProcessor
from climate_zarr.climate_config import get_config

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
        output = Path(Prompt.ask(
            "📊 Output CSV file",
            default=f"{region}_{variable}_stats.csv"
        ))
    elif not output:
        output = Path("county_stats.csv")
    
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
            n_workers=workers,
            memory_limit="4GB",
            use_distributed=use_distributed
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
            threshold_mm=threshold,
            chunk_by_county=chunk_by_county
        )
        
        # Save results
        results_df.to_csv(output, index=False)
        
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
    
    # Step 2: File selection
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
        console.print("\n[bold blue]📈 Step 3: Configure county statistics[/bold blue]")
        
        # Use existing zarr or ask for path
        if operation == "stats":
            zarr_path = Path(questionary.path("📁 Path to Zarr dataset:").ask())
            if not zarr_path.exists():
                console.print(f"[red]❌ Zarr dataset not found: {zarr_path}[/red]")
                return
        else:
            zarr_path = output_path
        
        # Region for statistics
        stats_region = interactive_region_selection()
        
        # Variable selection
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
        
        # Output file
        output_csv = Path(questionary.text(
            "📊 Output CSV file name:",
            default=f"{stats_region}_{variable}_stats.csv"
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
                n_workers=int(workers),
                memory_limit="4GB",
                use_distributed=use_distributed
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
                threshold_mm=float(threshold),
                chunk_by_county=True
            )
            
            # Save results
            results_df.to_csv(output_csv, index=False)
            
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


if __name__ == "__main__":
    app() 