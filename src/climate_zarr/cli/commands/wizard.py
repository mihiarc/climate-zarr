#!/usr/bin/env python
"""Interactive wizard command (skeleton)."""

import typer
from rich.console import Console

console = Console(highlight=False)
app = typer.Typer(add_completion=False, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def run():
    """Start the interactive wizard (placeholder to keep command)."""
    console.print("Welcome to the Climate Data Processing Wizard")
    # For now, we just greet; full interactive flow can be ported incrementally

