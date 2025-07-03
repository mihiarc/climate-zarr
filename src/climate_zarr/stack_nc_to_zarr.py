#!/usr/bin/env python
"""Stack multiple NetCDF files into a single Zarr store."""

import argparse
from pathlib import Path
from typing import List, Optional
import xarray as xr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import numcodecs


console = Console()


# Import modern configuration
try:
    from climate_zarr.climate_config import get_config
    CONFIG = get_config()
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    # Fallback bounds for CONUS
    CONUS_BOUNDS = {
        'lat_min': 24.0,
        'lat_max': 50.0,
        'lon_min': -125.0,
        'lon_max': -66.0
    }


def clip_to_region(
    ds: xr.Dataset,
    bounds: dict,
    lat_dim: str = None,
    lon_dim: str = None
) -> xr.Dataset:
    """Clip dataset to a geographic region."""
    # Auto-detect lat/lon dimension names
    if lat_dim is None:
        lat_candidates = ['lat', 'latitude', 'Latitude', 'LAT']
        for candidate in lat_candidates:
            if candidate in ds.dims:
                lat_dim = candidate
                break
    
    if lon_dim is None:
        lon_candidates = ['lon', 'longitude', 'Longitude', 'LON']
        for candidate in lon_candidates:
            if candidate in ds.dims:
                lon_dim = candidate
                break
    
    if lat_dim is None or lon_dim is None:
        raise ValueError("Could not auto-detect latitude/longitude dimensions")
    
    # Get the coordinate arrays
    lons = ds[lon_dim].values
    
    # Handle different longitude conventions
    if lons.max() > 180:
        # Data is in 0-360 format, convert bounds to match
        lon_min_360 = bounds['lon_min'] if bounds['lon_min'] >= 0 else bounds['lon_min'] + 360
        lon_max_360 = bounds['lon_max'] if bounds['lon_max'] >= 0 else bounds['lon_max'] + 360
        
        # For CONUS, we need to handle the wrap-around
        if lon_min_360 > lon_max_360:  # Crosses 0/360 boundary
            # Select in two parts and concatenate
            ds_west = ds.sel({
                lat_dim: slice(bounds['lat_min'], bounds['lat_max']),
                lon_dim: slice(lon_min_360, 360)
            })
            ds_east = ds.sel({
                lat_dim: slice(bounds['lat_min'], bounds['lat_max']),
                lon_dim: slice(0, lon_max_360)
            })
            ds_clipped = xr.concat([ds_west, ds_east], dim=lon_dim)
        else:
            ds_clipped = ds.sel({
                lat_dim: slice(bounds['lat_min'], bounds['lat_max']),
                lon_dim: slice(lon_min_360, lon_max_360)
            })
    else:
        # Data is already in -180 to 180 format
        ds_clipped = ds.sel({
            lat_dim: slice(bounds['lat_min'], bounds['lat_max']),
            lon_dim: slice(bounds['lon_min'], bounds['lon_max'])
        })
    
    return ds_clipped


def stack_netcdf_to_zarr(
    nc_files: List[Path],
    zarr_path: Path,
    concat_dim: str = "time",
    chunks: Optional[dict] = None,
    compression: str = "default",
    compression_level: int = 5,
    clip_region: Optional[str] = None
) -> None:
    """Stack multiple NetCDF files into a single Zarr store."""
    
    console.print(f"[bold]Stacking {len(nc_files)} NetCDF files into Zarr[/bold]")
    
    # Open all datasets
    datasets = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading NetCDF files...", total=len(nc_files))
        
        for nc_file in sorted(nc_files):
            ds = xr.open_dataset(nc_file)
            
            # Clip to region if specified
            if clip_region:
                if HAS_CONFIG:
                    try:
                        region_config = CONFIG.get_region(clip_region)
                        bounds = {
                            'lat_min': region_config.lat_min,
                            'lat_max': region_config.lat_max,
                            'lon_min': region_config.lon_min,
                            'lon_max': region_config.lon_max
                        }
                        ds = clip_to_region(ds, bounds)
                    except ValueError:
                        console.print(f"[red]Unknown region: {clip_region}[/red]")
                        console.print(f"[yellow]Available regions: {list(CONFIG.regions.keys())}[/yellow]")
                        continue
                elif clip_region.lower() == 'conus':
                    ds = clip_to_region(ds, CONUS_BOUNDS)
                else:
                    console.print(f"[red]Region {clip_region} not supported without modern config[/red]")
            
            datasets.append(ds)
            progress.advance(task)
    
    # Concatenate along time dimension
    console.print(f"[blue]Concatenating along '{concat_dim}' dimension...[/blue]")
    combined_ds = xr.concat(datasets, dim=concat_dim)
    
    # Close individual datasets
    for ds in datasets:
        ds.close()
    
    # Apply chunking
    if chunks:
        combined_ds = combined_ds.chunk(chunks)
    else:
        # Use larger chunks for better compression with multiple years
        default_chunks = {}
        for dim in combined_ds.sizes:
            if dim == 'time':
                # Chunk by year (365 days) or total size if smaller
                default_chunks[dim] = min(365, combined_ds.sizes[dim])
            elif dim in ['lat', 'latitude']:
                default_chunks[dim] = min(180, combined_ds.sizes[dim])
            elif dim in ['lon', 'longitude']:
                default_chunks[dim] = min(360, combined_ds.sizes[dim])
            else:
                default_chunks[dim] = min(100, combined_ds.sizes[dim])
        combined_ds = combined_ds.chunk(default_chunks)
    
    # Set up compression
    if compression == "default":
        compressor = numcodecs.Blosc(cname='zstd', clevel=compression_level, shuffle=numcodecs.Blosc.SHUFFLE)
    elif compression == "zlib":
        compressor = numcodecs.Zlib(level=compression_level)
    elif compression == "gzip":
        compressor = numcodecs.GZip(level=compression_level)
    else:
        compressor = None
    
    # Apply compression to all data variables
    encoding = {}
    if compressor:
        for var in combined_ds.data_vars:
            encoding[var] = {'compressor': compressor}
    
    # Save to Zarr
    console.print("[blue]Writing to Zarr format...[/blue]")
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    combined_ds.to_zarr(zarr_path, mode='w', encoding=encoding, zarr_format=2)
    combined_ds.close()
    
    # Report sizes
    total_nc_size = sum(f.stat().st_size for f in nc_files) / (1024**2)  # MB
    console.print(f"\n[green]✓ Successfully stacked {len(nc_files)} files[/green]")
    console.print(f"Total NetCDF size: {total_nc_size:.1f} MB")
    console.print(f"Output: {zarr_path}")
    if clip_region:
        console.print(f"[yellow]Data clipped to {clip_region.upper()} region[/yellow]")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Stack multiple NetCDF files into Zarr")
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="NetCDF files to stack (or directory containing .nc files)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output Zarr store path"
    )
    parser.add_argument(
        "-d", "--concat-dim",
        type=str,
        default="time",
        help="Dimension to concatenate along (default: time)"
    )
    parser.add_argument(
        "-c", "--chunks",
        type=str,
        help="Chunk sizes as comma-separated key=value pairs"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="default",
        choices=["default", "zlib", "gzip", "none"],
        help="Compression algorithm (default: blosc/zstd)"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Compression level (default: 5)"
    )
    if HAS_CONFIG:
        available_regions = list(CONFIG.regions.keys())
        parser.add_argument(
            "--clip",
            type=str,
            choices=available_regions,
            help=f"Clip to a specific region. Available: {', '.join(available_regions)}"
        )
    else:
        parser.add_argument(
            "--clip",
            type=str,
            choices=['conus'],
            help="Clip to a specific region (e.g., 'conus' for Continental US)"
        )
    
    args = parser.parse_args()
    
    # Collect files
    nc_files = []
    for path in args.files:
        if path.is_dir():
            nc_files.extend(sorted(path.glob("*.nc")))
        elif path.exists():
            nc_files.append(path)
    
    if not nc_files:
        console.print("[red]No NetCDF files found![/red]")
        return
    
    # Parse chunks
    chunks = None
    if args.chunks:
        chunks = {}
        for chunk in args.chunks.split(','):
            key, value = chunk.split('=')
            chunks[key.strip()] = int(value.strip())
    
    # Stack files
    stack_netcdf_to_zarr(
        nc_files,
        args.output,
        args.concat_dim,
        chunks,
        args.compression,
        args.compression_level,
        args.clip
    )


if __name__ == "__main__":
    main()