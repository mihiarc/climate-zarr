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

from parallel_processor import ParallelClimateProcessor
from optimized_climate_calculator import OptimizedClimateCalculator


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
        calculator = OptimizedClimateCalculator(
            base_data_path=base_data_path,
            baseline_period=(1980, 2010),
            cache_dir=cache_dir,
            enable_caching=True
        )
        
        start = time.time()
        baseline = calculator.calculate_baseline_percentiles(county_info['bounds'])
        elapsed = time.time() - start
        
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
    processor = ParallelClimateProcessor(shapefile_path, base_data_path)
    
    if county_subset:
        counties = processor.counties[processor.counties['GEOID'].isin(county_subset)]
    else:
        counties = processor.counties
    
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
    from optimized_climate_calculator import OptimizedClimateCalculator
    
    processor = ParallelClimateProcessor(shapefile_path, "/dummy/path")
    calculator = OptimizedClimateCalculator(
        base_data_path="/dummy/path",
        cache_dir=cache_dir,
        enable_caching=True
    )
    
    cached = 0
    not_cached = 0
    
    for idx, county in processor.counties.iterrows():
        county_info = processor.prepare_county_info(county)
        cache_key = calculator._get_cache_key(county_info['bounds'], 'baseline_percentiles')
        
        if calculator._load_from_cache(cache_key) is not None:
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
    parser.add_argument('--shapefile', required=True, help='Path to county shapefile')
    parser.add_argument('--data-path', required=True, help='Path to climate data')
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