#!/usr/bin/env python
"""County stats command (modularized)."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from climate_zarr.county_processor import ModernCountyProcessor

console = Console(highlight=False)
app = typer.Typer(add_completion=False, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def run(
    zarr_path: Path = typer.Argument(..., help="Path to Zarr dataset"),
    region: str = typer.Argument(..., help="Region name (conus, alaska, hawaii, etc.)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV file"),
    variable: str = typer.Option("pr", "--variable", "-v", help="Climate variable to analyze"),
    scenario: str = typer.Option("historical", "--scenario", "-s", help="Scenario name"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Threshold value"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of worker processes"),
    chunk_by_county: bool = typer.Option(True, "--chunk-counties", help="Process counties in chunks"),
):
    """Calculate climate statistics by county for a specific region."""
    if not zarr_path.exists():
        console.print(f"[red]❌ Zarr dataset not found: {zarr_path}[/red]")
        raise typer.Exit(1)

    # Determine shapefile
    shapefile_name = {
        "conus": "conus_counties.shp",
        "alaska": "alaska_counties.shp",
        "hawaii": "hawaii_counties.shp",
        "guam": "guam_counties.shp",
        "puerto_rico": "puerto_rico_counties.shp",
    }.get(region, f"{region}_counties.shp")
    shapefile_path = Path("regional_counties") / shapefile_name
    if not shapefile_path.exists():
        console.print(f"[red]❌ Shapefile not found: {shapefile_path}[/red]")
        raise typer.Exit(1)

    with ModernCountyProcessor(n_workers=workers) as processor:
        gdf = processor.prepare_shapefile(shapefile_path)
        results_df = processor.process_zarr_data(
            zarr_path=zarr_path,
            gdf=gdf,
            scenario=scenario,
            variable=variable,
            threshold=threshold if threshold is not None else (25.4 if variable == "pr" else 0),
            chunk_by_county=chunk_by_county,
        )

    if output is None:
        output = Path("climate_outputs/stats") / variable / region / scenario / f"{region}_{scenario}_{variable}_stats.csv"
        output.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output, index=False)
    console.print(Panel.fit(f"[green]✅ Saved results to {output}[/green]", border_style="green"))

