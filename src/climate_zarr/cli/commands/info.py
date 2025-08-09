#!/usr/bin/env python
"""Info command (modularized)."""

import typer
from rich.console import Console
from rich.panel import Panel

from climate_zarr.climate_config import get_config

console = Console(highlight=False)
app = typer.Typer(add_completion=False, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def run():
    """Display configuration and available data locations."""
    cfg = get_config()
    console.print(Panel.fit("[bold blue]Climate Zarr Toolkit[/bold blue]", border_style="blue"))
    console.print("[cyan]Available Data[/cyan]")
    console.print(f"Output base: {cfg.output.base_output_dir}")

