#!/usr/bin/env python
"""List regions command (modularized)."""

import typer
from rich.console import Console
from rich.table import Table

from climate_zarr.climate_config import get_config

console = Console(highlight=False)
app = typer.Typer(add_completion=False, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def run():
    """List available regions from configuration."""
    cfg = get_config()
    table = Table(title="Available Regions")
    table.add_column("Key", style="cyan")
    table.add_column("Name", style="white")
    for key, rc in cfg.regions.items():
        table.add_row(key, rc.name)
    console.print(table)

