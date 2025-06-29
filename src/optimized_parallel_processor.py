#!/usr/bin/env python3
"""
Optimized parallel processor using caching and improved I/O strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Callable
import time

from parallel_processor import ParallelClimateProcessor
from optimized_climate_calculator import OptimizedClimateCalculator, BatchOptimizedCalculator


class OptimizedParallelProcessor(ParallelClimateProcessor):
    """
    Optimized parallel processor with caching and performance improvements.
    """
    
    def __init__(self, 
                 counties_shapefile_path: str,
                 base_data_path: str,
                 baseline_period: Tuple[int, int] = (1980, 2010),
                 cache_dir: Optional[str] = None,
                 enable_caching: bool = True):
        """
        Initialize optimized parallel processor.
        
        Parameters
        ----------
        cache_dir : str, optional
            Directory for cache files
        enable_caching : bool
            Whether to enable baseline caching
        """
        super().__init__(counties_shapefile_path, base_data_path, baseline_period)
        self.cache_dir = cache_dir
        self.enable_caching = enable_caching
        
    @staticmethod
    def process_county_batch_optimized(batch_info: Dict) -> List[Dict]:
        """
        Optimized batch processing with caching and better I/O.
        """
        # Create optimized calculator in worker process
        calculator = BatchOptimizedCalculator(
            base_data_path=batch_info['base_data_path'],
            baseline_period=batch_info['baseline_period'],
            cache_dir=batch_info.get('cache_dir'),
            enable_caching=batch_info.get('enable_caching', True)
        )
        
        batch_id = batch_info['batch_id']
        counties = batch_info['counties']
        
        print(f"Worker {batch_id}: Processing {len(counties)} counties with optimizations")
        start_time = time.time()
        
        # Use batch processing for shared file handles
        results = calculator.process_county_batch(
            county_infos=counties,
            scenarios=batch_info['scenarios'],
            variables=batch_info['variables'],
            historical_period=batch_info['historical_period'],
            future_period=batch_info['future_period']
        )
        
        elapsed = time.time() - start_time
        counties_per_second = len(counties) / elapsed if elapsed > 0 else 0
        
        print(f"Worker {batch_id}: Completed in {elapsed:.1f}s ({counties_per_second:.2f} counties/s)")
        
        return results
    
    def create_optimized_batches(self, 
                               counties_subset: Optional[pd.DataFrame] = None,
                               target_counties_per_batch: int = 50) -> List[List[Dict]]:
        """
        Create optimized batches for processing.
        
        Uses larger batches to maximize file handle sharing.
        """
        counties_to_process = counties_subset if counties_subset is not None else self.counties
        
        # Prepare county info
        county_infos = [
            self.prepare_county_info(row) 
            for _, row in counties_to_process.iterrows()
        ]
        
        # Create batches
        batches = []
        for i in range(0, len(county_infos), target_counties_per_batch):
            batches.append(county_infos[i:i + target_counties_per_batch])
            
        return batches
    
    def process_parallel_optimized(self,
                                 scenarios: List[str] = ['historical', 'ssp245'],
                                 variables: List[str] = ['tas', 'tasmax', 'tasmin', 'pr'],
                                 historical_period: Tuple[int, int] = (1980, 2010),
                                 future_period: Tuple[int, int] = (2040, 2070),
                                 n_workers: Optional[int] = None,
                                 counties_subset: Optional[pd.DataFrame] = None,
                                 progress_callback: Optional[Callable] = None,
                                 counties_per_batch: int = 50) -> pd.DataFrame:
        """
        Optimized parallel processing with caching and better batching.
        """
        if n_workers is None:
            n_workers = min(mp.cpu_count() - 1, 16)
            
        # Create optimized batches
        batches = self.create_optimized_batches(counties_subset, counties_per_batch)
        n_counties = sum(len(batch) for batch in batches)
        
        print(f"\nOptimized Parallel Processing:")
        print(f"  Total counties: {n_counties}")
        print(f"  Counties per batch: {counties_per_batch}")
        print(f"  Number of batches: {len(batches)}")
        print(f"  Worker processes: {n_workers}")
        print(f"  Caching enabled: {self.enable_caching}")
        
        # Prepare batch info
        batch_infos = []
        for i, batch in enumerate(batches):
            batch_info = {
                'counties': batch,
                'base_data_path': str(self.base_data_path),
                'baseline_period': self.baseline_period,
                'scenarios': scenarios,
                'variables': variables,
                'historical_period': historical_period,
                'future_period': future_period,
                'batch_id': i,
                'cache_dir': self.cache_dir,
                'enable_caching': self.enable_caching
            }
            batch_infos.append(batch_info)
            
        # Process batches
        all_results = []
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit batches
            future_to_batch = {
                executor.submit(self.process_county_batch_optimized, batch_info): i 
                for i, batch_info in enumerate(batch_infos)
            }
            
            # Process completed batches
            completed = 0
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    completed += 1
                    
                    # Progress update
                    progress = (completed / len(batches)) * 100
                    elapsed = time.time() - start_time
                    
                    if progress_callback:
                        progress_callback(completed, len(batches), elapsed)
                        
                except Exception as e:
                    print(f"\nERROR in batch {batch_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
        # Summary
        total_time = time.time() - start_time
        counties_per_second = n_counties / total_time if total_time > 0 else 0
        
        print(f"\nOptimized Processing Complete:")
        print(f"  Total time: {total_time:.1f} seconds")
        print(f"  Average: {counties_per_second:.2f} counties/second")
        print(f"  Records generated: {len(all_results)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            df = df.sort_values(['GEOID', 'scenario', 'year'])
            
        return df


def compare_performance(shapefile_path: str, base_data_path: str, 
                       n_test_counties: int = 20) -> Dict:
    """
    Compare performance of original vs optimized processors.
    """
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Test parameters
    test_scenarios = ['historical']
    test_period = (2005, 2010)
    n_workers = min(mp.cpu_count() // 2, 8)
    
    results = {}
    
    # 1. Original processor
    print("\n1. Original Parallel Processor:")
    original = ParallelClimateProcessor(shapefile_path, base_data_path)
    test_counties = original.counties.sample(n=n_test_counties)
    
    start = time.time()
    df_original = original.process_parallel(
        counties_subset=test_counties,
        scenarios=test_scenarios,
        historical_period=test_period,
        future_period=(2040, 2041),
        n_workers=n_workers
    )
    original_time = time.time() - start
    
    results['original'] = {
        'time': original_time,
        'counties_per_second': n_test_counties / original_time,
        'records': len(df_original)
    }
    
    # 2. Optimized processor (no cache)
    print("\n2. Optimized Processor (no cache):")
    optimized_no_cache = OptimizedParallelProcessor(
        shapefile_path, base_data_path, enable_caching=False
    )
    
    start = time.time()
    df_opt_no_cache = optimized_no_cache.process_parallel_optimized(
        counties_subset=test_counties,
        scenarios=test_scenarios,
        historical_period=test_period,
        future_period=(2040, 2041),
        n_workers=n_workers,
        counties_per_batch=10
    )
    opt_no_cache_time = time.time() - start
    
    results['optimized_no_cache'] = {
        'time': opt_no_cache_time,
        'counties_per_second': n_test_counties / opt_no_cache_time,
        'records': len(df_opt_no_cache),
        'speedup': original_time / opt_no_cache_time
    }
    
    # 3. Optimized processor (with cache) - Run twice
    print("\n3. Optimized Processor (with cache) - First run:")
    optimized = OptimizedParallelProcessor(
        shapefile_path, base_data_path, enable_caching=True
    )
    
    start = time.time()
    df_opt1 = optimized.process_parallel_optimized(
        counties_subset=test_counties,
        scenarios=test_scenarios,
        historical_period=test_period,
        future_period=(2040, 2041),
        n_workers=n_workers,
        counties_per_batch=10
    )
    opt_cache_time1 = time.time() - start
    
    print("\n4. Optimized Processor (with cache) - Second run:")
    start = time.time()
    df_opt2 = optimized.process_parallel_optimized(
        counties_subset=test_counties,
        scenarios=test_scenarios,
        historical_period=test_period,
        future_period=(2040, 2041),
        n_workers=n_workers,
        counties_per_batch=10
    )
    opt_cache_time2 = time.time() - start
    
    results['optimized_cache_first'] = {
        'time': opt_cache_time1,
        'counties_per_second': n_test_counties / opt_cache_time1,
        'speedup': original_time / opt_cache_time1
    }
    
    results['optimized_cache_second'] = {
        'time': opt_cache_time2,
        'counties_per_second': n_test_counties / opt_cache_time2,
        'speedup': original_time / opt_cache_time2
    }
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"\nTest configuration:")
    print(f"  Counties: {n_test_counties}")
    print(f"  Workers: {n_workers}")
    print(f"  Period: {test_period[0]}-{test_period[1]}")
    
    print(f"\nResults:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Time: {result['time']:.2f}s")
        print(f"  Counties/second: {result['counties_per_second']:.2f}")
        if 'speedup' in result:
            print(f"  Speedup vs original: {result['speedup']:.1f}x")
            
    return results


# Example usage
if __name__ == "__main__":
    # Run performance comparison
    results = compare_performance(
        shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        n_test_counties=20
    )
    
    # Estimate full scale with optimizations
    print("\n" + "="*60)
    print("FULL SCALE ESTIMATES WITH OPTIMIZATIONS")
    print("="*60)
    
    if 'optimized_cache_second' in results:
        counties_per_second = results['optimized_cache_second']['counties_per_second']
        total_counties = 3235  # Approximate US counties
        
        print(f"\nWith optimized processing at {counties_per_second:.2f} counties/second:")
        
        for n_workers in [8, 16, 32, 64]:
            # Assume 80% efficiency for scaling
            effective_rate = counties_per_second * n_workers * 0.8
            total_hours = total_counties / effective_rate / 3600
            
            print(f"  {n_workers} workers: {total_hours:.1f} hours ({total_hours/24:.1f} days)")