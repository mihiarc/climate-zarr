#!/usr/bin/env python
"""Typer application aggregator for Climate Zarr CLI."""

import typer

from .commands.create_zarr import app as create_zarr_app
from .commands.county_stats import app as county_stats_app
from .commands.info import app as info_app
from .commands.regions import app as regions_app
from .commands.wizard import app as wizard_app

app = typer.Typer(
    name="climate-zarr",
    help="🌡️ Modern climate data processing toolkit",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Mount sub-apps under main app
app.add_typer(create_zarr_app, name="create-zarr")
app.add_typer(county_stats_app, name="county-stats")
app.add_typer(info_app, name="info")
app.add_typer(regions_app, name="list-regions")
app.add_typer(wizard_app, name="wizard")

