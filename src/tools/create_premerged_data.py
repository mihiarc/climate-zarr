#!/usr/bin/env python3
"""
Pre-merge climate data for optimized county processing.

This script creates pre-merged climate data files organized by:
1. Time-merged baseline files (1980-2010) for percentile calculations
2. County-extracted data subsets for faster processing
3. Regional tiles covering multiple counties

Benefits:
- Eliminates redundant file I/O (currently each county loads same files)
- Reduces data volume by pre-extracting county regions
- Enables parallel processing without file contention
"""

import sys
import time
import pickle
import json
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.unified_processor import UnifiedParallelProcessor
from src.archive.optimized_climate_calculator import OptimizedClimateCalculator


class ClimateDataPreMerger:
    """Pre-merge climate data for optimized processing."""
    
    def __init__(self, base_data_path: str, output_dir: str, shapefile_path: str):
        self.base_data_path = Path(base_data_path)
        self.output_dir = Path(output_dir)
        self.shapefile_path = shapefile_path
        
        # Load county information
        self.counties_gdf = gpd.read_file(shapefile_path)
        self.counties_gdf = self.counties_gdf.to_crs('EPSG:4326')
        
        print(f"Loaded {len(self.counties_gdf)} counties")
        
    def create_baseline_merged_files(self, variables: List[str] = ['tasmax', 'tasmin']):
        """Create time-merged baseline files (1980-2010) for each variable."""
        
        baseline_dir = self.output_dir / 'baseline_merged'
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        for variable in variables:
            print(f"\nMerging baseline files for {variable}...")
            
            # Find all baseline year files
            historical_dir = self.base_data_path / variable / 'historical'
            if not historical_dir.exists():
                print(f"  Historical directory not found: {historical_dir}")
                continue
            
            # Collect baseline files (1980-2010)
            baseline_files = []
            for year in range(1980, 2011):
                year_files = list(historical_dir.glob(f"{variable}_day_*_historical_*_{year}.nc"))
                baseline_files.extend(year_files)
            
            if not baseline_files:
                print(f"  No baseline files found")
                continue
            
            print(f"  Found {len(baseline_files)} files to merge")
            output_file = baseline_dir / f"{variable}_baseline_1980-2010.nc"
            
            if output_file.exists():
                print(f"  Output already exists: {output_file}")
                continue
            
            # Merge files
            start_time = time.time()
            
            # Open all files with dask for memory efficiency
            print(f"  Opening files with xarray...")
            ds = xr.open_mfdataset(
                baseline_files,
                combine='by_coords',
                chunks={'time': 365, 'lat': 100, 'lon': 100}
            )
            
            # Sort by time
            ds = ds.sortby('time')
            
            # Save with compression
            print(f"  Writing merged file...")
            encoding = {
                variable: {
                    'zlib': True,
                    'complevel': 4,
                    'shuffle': True,
                    'chunksizes': (365, 100, 100)
                }
            }
            
            ds.to_netcdf(output_file, encoding=encoding)
            
            elapsed = time.time() - start_time
            size_mb = output_file.stat().st_size / (1024 * 1024)
            
            print(f"  ✓ Created {output_file.name}")
            print(f"    Time: {elapsed:.1f}s")
            print(f"    Size: {size_mb:.1f} MB")
            
            ds.close()
    
    def create_county_extracts(self, 
                              counties_subset: Optional[List[str]] = None,
                              variables: List[str] = ['tasmax', 'tasmin', 'pr'],
                              scenarios: List[str] = ['historical', 'ssp245', 'ssp585'],
                              year_range: Tuple[int, int] = (2015, 2050)):
        """Pre-extract data for each county to eliminate redundant region selection."""
        
        county_dir = self.output_dir / 'county_extracts'
        county_dir.mkdir(parents=True, exist_ok=True)
        
        # Select counties to process
        if counties_subset:
            counties_to_process = self.counties_gdf[
                self.counties_gdf['GEOID'].isin(counties_subset)
            ]
        else:
            counties_to_process = self.counties_gdf
        
        print(f"\nCreating county extracts for {len(counties_to_process)} counties")
        
        # Process each county
        for idx, county in counties_to_process.iterrows():
            geoid = county['GEOID']
            name = county['NAME']
            
            county_output_dir = county_dir / geoid
            
            # Skip if already processed
            if county_output_dir.exists() and len(list(county_output_dir.glob("*.nc"))) > 0:
                print(f"  Skipping {geoid} {name} (already processed)")
                continue
            
            county_output_dir.mkdir(exist_ok=True)
            
            print(f"\n  Processing {geoid} {name}...")
            
            # Get county bounds
            bounds = county.geometry.bounds  # (minx, miny, maxx, maxy)
            min_lon, min_lat, max_lon, max_lat = bounds
            
            # Add buffer for edge effects
            buffer = 0.5
            min_lon -= buffer
            max_lon += buffer  
            min_lat -= buffer
            max_lat += buffer
            
            # Process each variable and scenario
            for variable in variables:
                for scenario in scenarios:
                    self._extract_county_data(
                        geoid, variable, scenario, year_range,
                        (min_lon, min_lat, max_lon, max_lat),
                        county_output_dir
                    )
            
            # Save county metadata
            metadata = {
                'geoid': geoid,
                'name': name,
                'state': county['STATEFP'],
                'bounds': list(bounds),
                'buffered_bounds': [min_lon, min_lat, max_lon, max_lat],
                'variables': variables,
                'scenarios': scenarios,
                'year_range': list(year_range)
            }
            
            with open(county_output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _extract_county_data(self, geoid: str, variable: str, scenario: str,
                           year_range: Tuple[int, int], bounds: Tuple[float, float, float, float],
                           output_dir: Path):
        """Extract data for a single county, variable, and scenario."""
        
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Adjust longitude if needed
        if min_lon < 0:
            min_lon = min_lon % 360
        if max_lon < 0:
            max_lon = max_lon % 360
        
        # Find files for this period
        scenario_dir = self.base_data_path / variable / scenario
        if not scenario_dir.exists():
            return
        
        # Collect files for year range
        files = []
        for year in range(year_range[0], year_range[1] + 1):
            year_files = list(scenario_dir.glob(f"{variable}_day_*_{scenario}_*_{year}.nc"))
            files.extend(year_files)
        
        if not files:
            return
        
        output_file = output_dir / f"{variable}_{scenario}_{year_range[0]}-{year_range[1]}.nc"
        
        if output_file.exists():
            return
        
        # Extract and save
        try:
            # Open files
            ds = xr.open_mfdataset(files, combine='by_coords')
            
            # Extract region
            regional = ds[variable].sel(
                lat=slice(min_lat, max_lat),
                lon=slice(min_lon, max_lon)
            )
            
            # Create dataset with metadata
            regional_ds = xr.Dataset({variable: regional})
            regional_ds.attrs = {
                'geoid': geoid,
                'variable': variable,
                'scenario': scenario,
                'original_bounds': list(bounds),
                'extracted': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save with compression
            encoding = {
                variable: {
                    'zlib': True,
                    'complevel': 4,
                    'shuffle': True
                }
            }
            
            regional_ds.to_netcdf(output_file, encoding=encoding)
            
        except Exception as e:
            print(f"    Error extracting {variable} {scenario}: {e}")
    
    def create_regional_tiles(self, tile_size: float = 2.0):
        """Create regional tiles that cover multiple counties for shared processing."""
        
        tiles_dir = self.output_dir / 'regional_tiles'
        tiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Get overall bounds
        total_bounds = self.counties_gdf.total_bounds  # [minx, miny, maxx, maxy]
        min_lon, min_lat, max_lon, max_lat = total_bounds
        
        # Create tile grid
        lon_tiles = np.arange(min_lon, max_lon, tile_size)
        lat_tiles = np.arange(min_lat, max_lat, tile_size)
        
        print(f"\nCreating regional tiles ({len(lon_tiles)} x {len(lat_tiles)})")
        
        # Create spatial index for counties
        counties_sindex = self.counties_gdf.sindex
        
        # Create tiles
        tile_info = {}
        tile_id = 0
        
        for i, lon_start in enumerate(lon_tiles):
            for j, lat_start in enumerate(lat_tiles):
                lon_end = min(lon_start + tile_size, max_lon)
                lat_end = min(lat_start + tile_size, max_lat)
                
                tile_bounds = (lon_start, lat_start, lon_end, lat_end)
                
                # Find counties in this tile
                possible_matches_index = list(counties_sindex.intersection(tile_bounds))
                possible_matches = self.counties_gdf.iloc[possible_matches_index]
                
                # Get counties that actually intersect
                from shapely.geometry import box
                tile_geom = box(*tile_bounds)
                intersecting = possible_matches[possible_matches.intersects(tile_geom)]
                
                if len(intersecting) > 0:
                    tile_name = f"tile_{tile_id:04d}"
                    tile_info[tile_name] = {
                        'bounds': list(tile_bounds),
                        'counties': intersecting['GEOID'].tolist(),
                        'n_counties': len(intersecting)
                    }
                    tile_id += 1
        
        # Save tile index
        with open(tiles_dir / 'tile_index.json', 'w') as f:
            json.dump(tile_info, f, indent=2)
        
        print(f"  Created {len(tile_info)} tiles covering counties")
        
        # Create tile-to-county mapping
        county_to_tiles = {}
        for tile_name, info in tile_info.items():
            for geoid in info['counties']:
                if geoid not in county_to_tiles:
                    county_to_tiles[geoid] = []
                county_to_tiles[geoid].append(tile_name)
        
        with open(tiles_dir / 'county_tile_mapping.json', 'w') as f:
            json.dump(county_to_tiles, f, indent=2)
        
        return tile_info
    
    def create_performance_report(self):
        """Generate report on pre-merge benefits."""
        
        report = {
            'pre_merge_benefits': {
                'baseline_merged': {
                    'description': 'Time-merged baseline files (1980-2010)',
                    'benefit': 'Eliminate loading 10,950 individual files per variable',
                    'speedup_estimate': '50-100x for baseline calculations'
                },
                'county_extracts': {
                    'description': 'Pre-extracted county-specific data',
                    'benefit': 'Eliminate redundant regional selection from global files',
                    'speedup_estimate': '5-10x for data loading',
                    'storage_overhead': 'Approx 10-50 MB per county depending on time range'
                },
                'regional_tiles': {
                    'description': 'Shared regional tiles for nearby counties',
                    'benefit': 'Process multiple counties from same data load',
                    'speedup_estimate': '2-5x for batch processing'
                }
            },
            'implementation_strategy': {
                'phase1': 'Use baseline_merged for percentile calculations',
                'phase2': 'Use county_extracts for scenario processing',
                'phase3': 'Use regional_tiles for batch optimization'
            }
        }
        
        report_file = self.output_dir / 'premerge_performance_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nPerformance report saved: {report_file}")
        
        return report


def main():
    """Main function to create pre-merged data."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-merge climate data for optimized processing')
    parser.add_argument('--data-path', 
                       default='/media/mihiarc/RPA1TB/CLIMATE_DATA/climate_baselines_backup',
                       help='Path to climate data directory')
    parser.add_argument('--shapefile', 
                       default='/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp',
                       help='Path to county shapefile')
    parser.add_argument('--output-dir', 
                       default='/media/mihiarc/RPA1TB/CLIMATE_DATA/merged_baselines',
                       help='Output directory for pre-merged data')
    
    # Pre-merge options
    parser.add_argument('--merge-baselines', action='store_true',
                       help='Create time-merged baseline files')
    parser.add_argument('--create-extracts', action='store_true',
                       help='Create county-specific data extracts')
    parser.add_argument('--create-tiles', action='store_true',
                       help='Create regional tiles')
    parser.add_argument('--counties', nargs='+',
                       help='Specific county GEOIDs to process (default: all)')
    parser.add_argument('--tile-size', type=float, default=2.0,
                       help='Tile size in degrees (default: 2.0)')
    
    args = parser.parse_args()
    
    print("CLIMATE DATA PRE-MERGER")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    
    # Create pre-merger
    merger = ClimateDataPreMerger(
        base_data_path=args.data_path,
        output_dir=args.output_dir,
        shapefile_path=args.shapefile
    )
    
    # Execute requested operations
    if args.merge_baselines:
        print("\nMERGING BASELINE FILES...")
        merger.create_baseline_merged_files()
    
    if args.create_extracts:
        print("\nCREATING COUNTY EXTRACTS...")
        merger.create_county_extracts(counties_subset=args.counties)
    
    if args.create_tiles:
        print("\nCREATING REGIONAL TILES...")
        merger.create_regional_tiles(tile_size=args.tile_size)
    
    # Generate report
    print("\nGENERATING PERFORMANCE REPORT...")
    merger.create_performance_report()
    
    print("\n" + "=" * 50)
    print("PRE-MERGE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()