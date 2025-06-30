#!/usr/bin/env python3
"""
Create organized backup of baseline cache files with county mapping.
"""

import sys
import shutil
import pickle
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.unified_processor import UnifiedParallelProcessor
from src.archive.optimized_climate_calculator import OptimizedClimateCalculator

def backup_baselines(shapefile_path, backup_dir, cache_dir=None):
    """Create organized backup of baseline cache files."""
    
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Load processor to get county info
    processor = UnifiedParallelProcessor(shapefile_path, "/dummy/path")
    calculator = OptimizedClimateCalculator(
        base_data_path="/dummy/path",
        cache_dir=cache_dir,
        enable_caching=True
    )
    
    # Create mapping of cache keys to county info
    mapping = {}
    cached_files = []
    
    print(f"Creating backup in: {backup_path}")
    print(f"Source cache dir: {calculator.cache_dir}")
    
    for idx, county in processor.counties.iterrows():
        county_info = processor.prepare_county_info(county)
        cache_key = calculator._get_cache_key(county_info['bounds'], 'baseline_percentiles')
        
        cache_file = calculator.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            # Copy file with descriptive name
            backup_file = backup_path / f"{county['GEOID']}_{county['NAME'].replace(' ', '_')}_{cache_key}.pkl"
            shutil.copy2(cache_file, backup_file)
            
            mapping[cache_key] = {
                'geoid': county['GEOID'],
                'name': county['NAME'],
                'state': county['STATEFP'],
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
        'total_counties': len(processor.counties),
        'cached_counties': len(cached_files),
        'backup_date': str(Path().resolve()),
        'cache_coverage': f"{len(cached_files)/len(processor.counties)*100:.1f}%"
    }
    
    summary_file = backup_path / "backup_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBackup complete:")
    print(f"  Cached counties: {len(cached_files)}/{len(processor.counties)}")
    print(f"  Coverage: {len(cached_files)/len(processor.counties)*100:.1f}%")
    print(f"  Files backed up: {len(cached_files)} + 2 metadata files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup baseline cache files')
    parser.add_argument('--shapefile', default='data/shapefiles/tl_2024_us_county.shp', 
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