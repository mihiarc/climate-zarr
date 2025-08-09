"""
CLI package for Climate Zarr.

This package organizes the Typer-based CLI into modular subcommands
to improve maintainability and testability.
"""

from .app import app  # re-export main Typer app

__all__ = ["app"]

