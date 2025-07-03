#!/usr/bin/env python
"""
🎮 Interactive Climate Zarr CLI Demo - 2025 Edition

This demo showcases the modern interactive features of the Climate Zarr CLI,
including guided wizards, beautiful prompts, and intelligent suggestions.

Features demonstrated:
- 🧙‍♂️ Interactive wizard for guided processing
- 🎯 Smart prompts and selection menus
- ✅ Confirmation dialogs for safety
- 📊 Real-time progress and beautiful output
- 🎨 Rich interface with modern UX patterns
"""

import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text

console = Console()

def print_section_header(title: str, description: str):
    """Print a beautiful section header."""
    console.print()
    console.print("─" * 80, style="dim")
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print(f"[dim]{description}[/dim]")
    console.print("─" * 80, style="dim")
    console.print()

def run_command_demo(cmd, description, show_output=True):
    """Run a command and display it nicely."""
    console.print(Panel(
        f"[yellow]💻 Command:[/yellow] [white]{' '.join(cmd)}[/white]\n"
        f"[cyan]📝 Description:[/cyan] {description}",
        border_style="blue",
        padding=(1, 2)
    ))
    
    if show_output:
        console.print("[dim]Running command...[/dim]")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                console.print("[green]✅ Command completed successfully![/green]")
                if result.stdout:
                    console.print("\n[bold]Output:[/bold]")
                    console.print(result.stdout)
            else:
                console.print(f"[red]❌ Command failed with exit code {result.returncode}[/red]")
                if result.stderr:
                    console.print(f"[red]Error:[/red] {result.stderr}")
        except subprocess.TimeoutExpired:
            console.print("[yellow]⏰ Command timed out (demo timeout)[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error running command: {e}[/red]")
    else:
        console.print("[dim]ℹ️ Command ready to run (demo mode)[/dim]")
    
    console.print()

def show_interactive_features():
    """Showcase the interactive features of the CLI."""
    
    features_table = Table(title="🎮 Interactive Features", show_header=True)
    features_table.add_column("Feature", style="cyan")
    features_table.add_column("Description", style="green")
    features_table.add_column("Command", style="yellow")
    
    features_table.add_row(
        "🧙‍♂️ Interactive Wizard",
        "Complete guided experience",
        "climate-zarr wizard"
    )
    features_table.add_row(
        "🗜️ Smart Conversion",
        "Interactive NetCDF → Zarr with prompts",
        "climate-zarr create-zarr"
    )
    features_table.add_row(
        "📈 Guided Statistics",
        "County analysis with intelligent defaults",
        "climate-zarr county-stats"
    )
    features_table.add_row(
        "🗺️ Region Selection",
        "Beautiful region picker with descriptions",
        "--region flag (interactive)"
    )
    features_table.add_row(
        "🔬 Variable Picker",
        "Climate variable selection with tooltips",
        "--variable flag (interactive)"
    )
    features_table.add_row(
        "⚠️ Safety Confirmations", 
        "Prevents accidental data loss",
        "All destructive operations"
    )
    
    console.print(features_table)

def main():
    """Run the interactive CLI demonstration."""
    
    # Welcome banner
    console.print(Panel.fit(
        "[bold blue]🎮 Climate Zarr Interactive CLI Demo[/bold blue]\n"
        "[bold green]2025 Modern Edition[/bold green]\n\n"
        "[dim]Showcasing cutting-edge interactive CLI patterns[/dim]",
        border_style="blue"
    ))
    
    # Check if CLI exists
    cli_path = Path("climate_cli.py")
    if not cli_path.exists():
        console.print("[red]❌ CLI script not found. Please run from the project directory.[/red]")
        return
    
    # Show interactive features overview
    print_section_header(
        "🌟 Interactive Features Overview", 
        "Modern CLI patterns with beautiful UX"
    )
    show_interactive_features()
    
    # Demo 1: Help and Information Commands
    print_section_header(
        "📖 Demo 1: Help & Information", 
        "Discover what the CLI can do"
    )
    
    run_command_demo(
        ["python", "climate_cli.py", "--help"],
        "Show main help with all available commands",
        show_output=True
    )
    
    run_command_demo(
        ["python", "climate_cli.py", "info"],
        "Display system information and available data",
        show_output=True
    )
    
    run_command_demo(
        ["python", "climate_cli.py", "list-regions"],
        "List all available regions for analysis",
        show_output=True
    )
    
    # Demo 2: Interactive Command Help
    print_section_header(
        "🎯 Demo 2: Interactive Command Help",
        "See how each command supports interactive mode"
    )
    
    run_command_demo(
        ["python", "climate_cli.py", "create-zarr", "--help"],
        "Interactive NetCDF to Zarr conversion options",
        show_output=True
    )
    
    run_command_demo(
        ["python", "climate_cli.py", "county-stats", "--help"],
        "Interactive county statistics analysis options",
        show_output=True
    )
    
    # Demo 3: Interactive Wizard (Info Mode)
    print_section_header(
        "🧙‍♂️ Demo 3: Interactive Wizard Preview",
        "See the wizard in action (info mode to avoid long processing)"
    )
    
    console.print(Panel(
        "[cyan]The wizard provides:[/cyan]\n"
        "• 🎯 Step-by-step guidance through the entire process\n"
        "• 📂 Smart file and directory detection\n"
        "• 🗺️ Regional selection with visual descriptions\n"
        "• 🔬 Climate variable picker with explanations\n"
        "• ⚙️ Performance optimization suggestions\n"
        "• ✅ Safety confirmations before processing\n"
        "• 📊 Beautiful progress tracking and results\n\n"
        "[yellow]Try it yourself:[/yellow] [white]python climate_cli.py wizard[/white]",
        title="🧙‍♂️ Interactive Wizard",
        border_style="cyan"
    ))
    
    # Demo 4: Non-Interactive Mode
    print_section_header(
        "⚡ Demo 4: Command-Line Mode",
        "Traditional CLI usage for automation and scripts"
    )
    
    run_command_demo(
        ["python", "climate_cli.py", "create-zarr", "data/", "-o", "demo.zarr", "--region", "conus", "--interactive", "false"],
        "Non-interactive mode for scripts and automation",
        show_output=False
    )
    
    run_command_demo(
        ["python", "climate_cli.py", "county-stats", "demo.zarr", "conus", "-v", "pr", "-t", "25.4", "--interactive", "false"],
        "Batch processing without prompts",
        show_output=False
    )
    
    # Demo 5: Error Handling and Recovery
    print_section_header(
        "🛡️ Demo 5: Smart Error Handling",
        "Intelligent error recovery and suggestions"
    )
    
    run_command_demo(
        ["python", "climate_cli.py", "create-zarr", "/nonexistent/path"],
        "Handle missing files with helpful suggestions",
        show_output=True
    )
    
    run_command_demo(
        ["python", "climate_cli.py", "county-stats", "missing.zarr", "invalid_region"],
        "Handle invalid parameters with interactive recovery",
        show_output=True
    )
    
    # Success summary
    print_section_header(
        "🎉 Demo Complete!",
        "You've seen the modern interactive CLI in action"
    )
    
    summary_panel = Panel(
        "[bold green]🎊 What You've Learned:[/bold green]\n\n"
        "✅ [cyan]Interactive Mode:[/cyan] Guided experience with intelligent prompts\n"
        "✅ [cyan]Command-Line Mode:[/cyan] Traditional CLI for automation\n"
        "✅ [cyan]Error Recovery:[/cyan] Smart suggestions when things go wrong\n"
        "✅ [cyan]Beautiful Output:[/cyan] Rich formatting and progress tracking\n"
        "✅ [cyan]Safety Features:[/cyan] Confirmations and validation\n\n"
        "[bold yellow]🚀 Ready to Get Started?[/bold yellow]\n\n"
        "[white]Try these commands:[/white]\n"
        "• [yellow]python climate_cli.py wizard[/yellow] - Full interactive experience\n"
        "• [yellow]python climate_cli.py create-zarr[/yellow] - Interactive conversion\n"
        "• [yellow]python climate_cli.py county-stats[/yellow] - Interactive analysis\n"
        "• [yellow]python climate_cli.py info[/yellow] - System overview\n\n"
        "[dim]💡 Pro tip: All commands work in both interactive and non-interactive modes![/dim]",
        title="🏆 Success",
        border_style="green",
        padding=(1, 2)
    )
    
    console.print(summary_panel)
    
    # Next steps
    console.print("\n" + "🌟" * 60)
    console.print("[bold blue]Ready to explore climate data with style! 🌡️[/bold blue]")
    console.print("🌟" * 60 + "\n")

if __name__ == "__main__":
    main() 