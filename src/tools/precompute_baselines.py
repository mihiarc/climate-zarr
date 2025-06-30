#!/usr/bin/env python3
"""
Pre-compute baseline percentiles for all US counties.

This script calculates and caches baseline percentiles for all counties,
eliminating the need to compute them during regular processing.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.unified_processor import UnifiedParallelProcessor
from src.core.unified_calculator import UnifiedClimateCalculator


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'baseline_computation_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


def compute_county_baseline(county_info, base_data_path, cache_dir):
    """Compute baseline for a single county."""
    try:
        calculator = UnifiedClimateCalculator(
            base_data_path=base_data_path,
            baseline_period=(1980, 2010)
        )
        
        start = time.time()
        # Calculate baseline percentiles directly
        baseline = calculator.calculate_baseline_percentiles_base(
            bounds=tuple(county_info['bounds']),
            variables=['tasmax', 'tasmin']
        )
        elapsed = time.time() - start
        
        # Save to cache if cache_dir provided
        if cache_dir and baseline:
            import hashlib
            import pickle
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Generate cache key matching the expected format
            bounds_tuple = tuple(county_info['bounds'])
            key_data = f"baseline_percentiles_{bounds_tuple}_{(1980, 2010)}".encode()
            cache_key = hashlib.md5(key_data).hexdigest()
            
            cache_file = cache_path / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(baseline, f)
        
        return {
            'status': 'success',
            'county': county_info['name'],
            'geoid': county_info['geoid'],
            'time': elapsed,
            'has_tasmax': 'tasmax_p90_doy' in baseline,
            'has_tasmin': 'tasmin_p10_doy' in baseline
        }
    except Exception as e:
        return {
            'status': 'error',
            'county': county_info['name'],
            'geoid': county_info['geoid'],
            'error': str(e)
        }


def precompute_all_baselines(shapefile_path, base_data_path, cache_dir=None, 
                           n_workers=None, county_subset=None):
    """Pre-compute baselines for all counties in parallel."""
    
    # Load counties
    processor = UnifiedParallelProcessor(shapefile_path, base_data_path)
    
    if county_subset:
        counties = processor.counties_gdf[processor.counties_gdf['GEOID'].isin(county_subset)]
    else:
        counties = processor.counties_gdf
    
    total_counties = len(counties)
    logging.info(f"Starting baseline computation for {total_counties} counties")
    
    # Prepare county info
    county_infos = []
    for idx, county in counties.iterrows():
        county_infos.append(processor.prepare_county_info(county))
    
    # Process in parallel
    if n_workers is None:
        n_workers = min(mp.cpu_count() - 1, 16)
    
    logging.info(f"Using {n_workers} worker processes")
    
    completed = 0
    successful = 0
    failed = 0
    total_time = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_county = {
            executor.submit(compute_county_baseline, county_info, base_data_path, cache_dir): county_info
            for county_info in county_infos
        }
        
        # Process results as they complete
        for future in as_completed(future_to_county):
            county_info = future_to_county[future]
            completed += 1
            
            try:
                result = future.result()
                
                if result['status'] == 'success':
                    successful += 1
                    total_time += result['time']
                    avg_time = total_time / successful
                    
                    logging.info(
                        f"[{completed}/{total_counties}] "
                        f"✓ {result['county']} ({result['geoid']}) "
                        f"- {result['time']:.1f}s"
                    )
                else:
                    failed += 1
                    logging.error(
                        f"[{completed}/{total_counties}] "
                        f"✗ {result['county']} ({result['geoid']}) "
                        f"- Error: {result['error']}"
                    )
                
                # Progress update every 10 counties
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total_counties - completed) / rate if rate > 0 else 0
                    
                    logging.info(
                        f"\nProgress: {completed}/{total_counties} ({completed/total_counties*100:.1f}%)"
                        f"\n  Successful: {successful}, Failed: {failed}"
                        f"\n  Average time per county: {avg_time:.1f}s"
                        f"\n  ETA: {eta/60:.1f} minutes"
                    )
                    
            except Exception as e:
                failed += 1
                logging.error(f"Failed to process {county_info['name']}: {str(e)}")
    
    # Final summary
    total_elapsed = time.time() - start_time
    logging.info("\n" + "="*60)
    logging.info("BASELINE COMPUTATION COMPLETE")
    logging.info("="*60)
    logging.info(f"Total counties: {total_counties}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logging.info(f"Average time per county: {total_time/successful:.1f}s")
    logging.info(f"Cache directory: {cache_dir or '~/.climate_cache'}")
    
    return successful, failed


def verify_cache_coverage(shapefile_path, cache_dir=None):
    """Verify which counties have cached baselines."""
    import hashlib
    
    processor = UnifiedParallelProcessor(shapefile_path, "/dummy/path")
    
    if cache_dir is None:
        cache_dir = Path.home() / '.climate_cache'
    else:
        cache_dir = Path(cache_dir)
    
    cached = 0
    not_cached = 0
    
    for idx, county in processor.counties_gdf.iterrows():
        county_info = processor.prepare_county_info(county)
        
        # Generate cache key matching the expected format
        bounds_tuple = tuple(county_info['bounds'])
        key_data = f"baseline_percentiles_{bounds_tuple}_{(1980, 2010)}".encode()
        cache_key = hashlib.md5(key_data).hexdigest()
        
        cache_file = cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cached += 1
        else:
            not_cached += 1
    
    total = cached + not_cached
    print(f"\nCache Coverage Report:")
    print(f"  Total counties: {total}")
    print(f"  Cached: {cached} ({cached/total*100:.1f}%)")
    print(f"  Not cached: {not_cached}")
    
    return cached, not_cached


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-compute baseline percentiles for all counties')
    parser.add_argument('--shapefile', 
                        default='/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp',
                        help='Path to county shapefile')
    parser.add_argument('--data-path', 
                        default='/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM',
                        help='Path to climate data')
    parser.add_argument('--cache-dir', help='Cache directory (default: ~/.climate_cache)')
    parser.add_argument('--workers', type=int, help='Number of workers (default: CPU count - 1)')
    parser.add_argument('--counties', nargs='+', help='Specific county GEOIDs to process')
    parser.add_argument('--verify', action='store_true', help='Only verify cache coverage')
    
    args = parser.parse_args()
    
    if args.verify:
        # Just check cache coverage
        verify_cache_coverage(args.shapefile, args.cache_dir)
    else:
        # Compute baselines
        precompute_all_baselines(
            shapefile_path=args.shapefile,
            base_data_path=args.data_path,
            cache_dir=args.cache_dir,
            n_workers=args.workers,
            county_subset=args.counties
        )