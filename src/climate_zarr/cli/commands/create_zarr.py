#!/usr/bin/env python
"""Create Zarr store command (modularized)."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm

from climate_zarr.stack_nc_to_zarr import stack_netcdf_to_zarr
from climate_zarr.climate_config import get_config

console = Console(highlight=False)
app = typer.Typer(add_completion=False, invoke_without_command=True)


def _print_banner():
    console.print(Panel.fit(
        "[bold blue]🌡️ Climate Zarr Toolkit[/bold blue]\n"
        "[dim]NetCDF → Zarr conversion[/dim]",
        border_style="blue",
    ))


@app.callback(invoke_without_command=True)
def run(
    input_path: Path = typer.Argument(..., help="Directory of NetCDF files or a single NetCDF file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output Zarr store path"),
    region: Optional[str] = typer.Option(None, "--region", "-r", help="Clip data to specific region"),
    concat_dim: str = typer.Option("time", "--concat-dim", "-d", help="Dimension to concatenate along"),
    chunks: Optional[str] = typer.Option(None, "--chunks", "-c", help="Chunk sizes 'dim1=size1,dim2=size2'"),
    compression: str = typer.Option("default", "--compression", help="Compression algorithm"),
    compression_level: int = typer.Option(5, "--compression-level", help="Compression level (1-9)"),
):
    """Convert NetCDF files to a Zarr store."""
    _print_banner()

    cfg = get_config()
    _ = cfg  # currently unused; kept for parity, future validation hooks

    if not input_path.exists():
        raise typer.Exit(code=1)

    # Collect NetCDF files
    if input_path.is_dir():
        nc_files = list(input_path.glob("*.nc"))
    elif input_path.is_file() and input_path.suffix == ".nc":
        nc_files = [input_path]
    else:
        nc_files = []

    if not nc_files:
        console.print(f"[red]❌ No NetCDF files found in: {input_path}[/red]")
        raise typer.Exit(1)

    if output is None:
        output = Path(f"{input_path.stem}_climate.zarr" if input_path.is_file() else "climate_data.zarr")

    # Parse chunks
    chunks_dict = None
    if chunks:
        chunks_dict = {}
        for chunk in chunks.split(","):
            key, value = chunk.split("=")
            chunks_dict[key.strip()] = int(value.strip())

    if len(nc_files) > 50:
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

    stack_netcdf_to_zarr(
        nc_files=nc_files,
        zarr_path=output,
        concat_dim=concat_dim,
        chunks=chunks_dict,
        compression=compression,
        compression_level=compression_level,
        clip_region=region,
    )

    console.print(Panel(f"[green]✅ Successfully created Zarr store: {output}[/green]", border_style="green"))

