#!/usr/bin/env python3
"""
Create organized backup of baseline cache files with county mapping.

This tool helps organize and backup baseline cache files from various sources.
"""

import sys
import shutil
import pickle
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.unified_processor import UnifiedParallelProcessor
from src.utils.state_fips import get_state_name


def get_cache_key(bounds, prefix='baseline', baseline_period=(1980, 2010)):
    """Generate a cache key from bounds - matching the original format."""
    # Convert bounds to tuple if it's a list
    if isinstance(bounds, list):
        bounds = tuple(bounds)
    # Match the original format: "prefix_(bound1, bound2, bound3, bound4)_(start, end)"
    key_data = f"{prefix}_{bounds}_{baseline_period}".encode()
    cache_key = hashlib.md5(key_data).hexdigest()
    return cache_key


def backup_baselines(shapefile_path, backup_dir, cache_dir=None):
    """Create organized backup of baseline cache files."""
    
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Load processor to get county info
    processor = UnifiedParallelProcessor(shapefile_path, "/dummy/path")
    
    # Default cache directory if not provided
    if cache_dir is None:
        cache_dir = Path.home() / '.climate_cache'
    else:
        cache_dir = Path(cache_dir)
    
    # Create mapping of cache keys to county info
    mapping = {}
    cached_files = []
    
    print(f"Creating backup in: {backup_path}")
    print(f"Source cache dir: {cache_dir}")
    
    # Check cache directory
    cache_files = list(cache_dir.glob("*.pkl"))
    print(f"Found {len(cache_files)} .pkl files in cache directory")
    
    for idx, county in processor.counties_gdf.iterrows():
        county_info = processor.prepare_county_info(county)
        # Try the most likely prefix first (baseline_percentiles)
        cache_key = get_cache_key(county_info['bounds'], prefix='baseline_percentiles')
        cache_file = cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            # Copy file with descriptive name
            backup_file = backup_path / f"{county['GEOID']}_{county['NAME'].replace(' ', '_')}_{cache_key}.pkl"
            shutil.copy2(cache_file, backup_file)
            
            mapping[cache_key] = {
                'geoid': county['GEOID'],
                'name': county['NAME'],
                'state': get_state_name(county['STATEFP']),
                'state_fips': county['STATEFP'],
                'bounds': county_info['bounds'],
                'cache_file': f"{county['GEOID']}_{county['NAME'].replace(' ', '_')}_{cache_key}.pkl",
                'original_file': f"{cache_key}.pkl"
            }
            cached_files.append(cache_key)
    
    # Save mapping file
    mapping_file = backup_path / "county_cache_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    # Save summary
    summary = {
        'total_counties': len(processor.counties_gdf),
        'cached_counties': len(cached_files),
        'backup_date': datetime.now().isoformat(),
        'cache_coverage': f"{len(cached_files)/len(processor.counties_gdf)*100:.1f}%",
        'cache_directory': str(cache_dir),
        'backup_directory': str(backup_path)
    }
    
    summary_file = backup_path / "backup_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBackup complete:")
    print(f"  Cached counties: {len(cached_files)}/{len(processor.counties_gdf)}")
    print(f"  Coverage: {len(cached_files)/len(processor.counties_gdf)*100:.1f}%")
    print(f"  Files backed up: {len(cached_files)} + 2 metadata files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup baseline cache files')
    parser.add_argument('--shapefile', default='/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp', 
                       help='Path to county shapefile')
    parser.add_argument('--backup-dir', default='/media/mihiarc/RPA1TB/CLIMATE_DATA/climate_baselines_backup',
                       help='Backup directory')
    parser.add_argument('--cache-dir', help='Cache directory (default: ~/.climate_cache)')
    
    args = parser.parse_args()
    
    backup_baselines(
        shapefile_path=args.shapefile,
        backup_dir=args.backup_dir,
        cache_dir=args.cache_dir
    )