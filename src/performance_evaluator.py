#!/usr/bin/env python3
"""
Performance evaluation and optimization strategies for climate data processing.
"""

import time
import psutil
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from climate_indicator_calculator import ClimateIndicatorCalculator
from parallel_processor import ParallelClimateProcessor


class PerformanceEvaluator:
    """
    Evaluates performance bottlenecks and tests optimization strategies.
    """
    
    def __init__(self, base_data_path: str, counties_shapefile_path: str):
        """Initialize performance evaluator."""
        self.base_data_path = Path(base_data_path)
        self.shapefile_path = Path(counties_shapefile_path)
        self.results = {}
        
    def profile_single_county(self, county_info: Dict, 
                            scenarios: List[str] = ['historical'],
                            years: Tuple[int, int] = (2009, 2010)) -> Dict:
        """
        Profile processing time for each step of single county processing.
        """
        print(f"\nProfiling {county_info['name']}...")
        timings = {}
        
        calculator = ClimateIndicatorCalculator(
            base_data_path=str(self.base_data_path),
            baseline_period=(1980, 2010)
        )
        
        # 1. Time baseline calculation
        start = time.time()
        thresholds = calculator.calculate_baseline_percentiles(county_info['bounds'])
        timings['baseline_calculation'] = time.time() - start
        print(f"  Baseline calculation: {timings['baseline_calculation']:.2f}s")
        
        # 2. Time data loading per variable
        variables = ['tas', 'tasmax', 'tasmin', 'pr']
        timings['data_loading'] = {}
        
        for var in variables:
            start = time.time()
            files = calculator.get_files_for_period(var, scenarios[0], years[0], years[1])
            if files:
                data = calculator.extract_county_data(files, var, county_info['bounds'])
            timings['data_loading'][var] = time.time() - start
            
        total_loading = sum(timings['data_loading'].values())
        print(f"  Data loading total: {total_loading:.2f}s")
        for var, t in timings['data_loading'].items():
            print(f"    {var}: {t:.2f}s")
        
        # 3. Time indicator calculation
        start = time.time()
        results = calculator.process_county(
            county_info=county_info,
            scenarios=scenarios,
            variables=variables,
            historical_period=years,
            future_period=(2040, 2041)
        )
        timings['total_processing'] = time.time() - start
        timings['indicator_calculation'] = timings['total_processing'] - total_loading - timings['baseline_calculation']
        
        print(f"  Indicator calculation: {timings['indicator_calculation']:.2f}s")
        print(f"  Total processing: {timings['total_processing']:.2f}s")
        
        # Memory usage
        process = psutil.Process()
        timings['memory_mb'] = process.memory_info().rss / 1024 / 1024
        print(f"  Memory usage: {timings['memory_mb']:.1f} MB")
        
        return timings
    
    def test_io_strategies(self, county_info: Dict) -> Dict:
        """
        Test different I/O strategies for NetCDF files.
        """
        print(f"\nTesting I/O strategies for {county_info['name']}...")
        results = {}
        
        # Get test files
        calculator = ClimateIndicatorCalculator(
            base_data_path=str(self.base_data_path),
            baseline_period=(1980, 2010)
        )
        files = calculator.get_files_for_period('tasmax', 'historical', 2000, 2010)[:5]
        
        if not files:
            print("  No files found for testing")
            return results
        
        bounds = county_info['bounds']
        
        # 1. Default xarray loading
        start = time.time()
        ds = xr.open_mfdataset(files, combine='by_coords')
        data = calculator.extract_county_data(files, 'tasmax', bounds)
        results['default_xarray'] = time.time() - start
        ds.close()
        
        # 2. With dask chunks
        start = time.time()
        ds = xr.open_mfdataset(files, combine='by_coords', chunks={'time': 365})
        data = calculator.extract_county_data(files, 'tasmax', bounds)
        results['dask_chunks'] = time.time() - start
        ds.close()
        
        # 3. Pre-select region before loading
        start = time.time()
        # Convert bounds to indices first
        min_lon, min_lat, max_lon, max_lat = bounds
        if min_lon < 0:
            min_lon = min_lon % 360
        if max_lon < 0:
            max_lon = max_lon % 360
            
        ds = xr.open_mfdataset(files, combine='by_coords')
        # Select region immediately
        regional = ds.sel(
            lat=slice(min_lat - 0.5, max_lat + 0.5),
            lon=slice(min_lon - 0.5, max_lon + 0.5)
        )
        data = regional['tasmax'].compute()
        results['preselect_region'] = time.time() - start
        ds.close()
        
        # 4. Load files sequentially
        start = time.time()
        arrays = []
        for f in files:
            ds = xr.open_dataset(f)
            regional = ds.sel(
                lat=slice(min_lat - 0.5, max_lat + 0.5),
                lon=slice(min_lon - 0.5, max_lon + 0.5)
            )
            arrays.append(regional['tasmax'])
            ds.close()
        combined = xr.concat(arrays, dim='time')
        results['sequential_loading'] = time.time() - start
        
        print("\nI/O Strategy Results:")
        for strategy, timing in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {strategy}: {timing:.3f}s")
            
        return results
    
    def test_parallelization_strategies(self, n_counties: int = 10) -> Dict:
        """
        Test different parallelization strategies.
        """
        print(f"\nTesting parallelization strategies with {n_counties} counties...")
        
        # Load processor and get test counties
        processor = ParallelClimateProcessor(
            counties_shapefile_path=str(self.shapefile_path),
            base_data_path=str(self.base_data_path)
        )
        
        test_counties = processor.counties.sample(n=min(n_counties, len(processor.counties)))
        test_period = (2009, 2010)  # Short period for testing
        
        results = {}
        
        # 1. Different numbers of workers
        for n_workers in [1, 2, 4, 8, 16]:
            if n_workers > mp.cpu_count():
                continue
                
            start = time.time()
            df = processor.process_parallel(
                counties_subset=test_counties,
                scenarios=['historical'],
                historical_period=test_period,
                future_period=(2040, 2041),
                n_workers=n_workers,
                variables=['tas', 'tasmax', 'tasmin', 'pr']
            )
            elapsed = time.time() - start
            
            results[f'workers_{n_workers}'] = {
                'time': elapsed,
                'counties_per_second': n_counties / elapsed,
                'records': len(df)
            }
            
            print(f"  {n_workers} workers: {elapsed:.1f}s ({n_counties/elapsed:.2f} counties/s)")
        
        # 2. Different batch sizes
        optimal_workers = min(mp.cpu_count() // 2, 8)
        for batch_factor in [1, 2, 4, 8]:
            batch_size = max(1, n_counties // (optimal_workers * batch_factor))
            
            batches = processor.create_batches(test_counties, batch_size)
            
            start = time.time()
            df = processor.process_parallel(
                counties_subset=test_counties,
                scenarios=['historical'],
                historical_period=test_period,
                future_period=(2040, 2041),
                n_workers=optimal_workers
            )
            elapsed = time.time() - start
            
            results[f'batch_size_{batch_size}'] = {
                'time': elapsed,
                'n_batches': len(batches),
                'counties_per_second': n_counties / elapsed
            }
            
            print(f"  Batch size {batch_size} ({len(batches)} batches): {elapsed:.1f}s")
            
        return results
    
    def test_caching_strategies(self, county_info: Dict) -> Dict:
        """
        Test impact of caching baseline percentiles.
        """
        print(f"\nTesting caching strategies...")
        results = {}
        
        calculator = ClimateIndicatorCalculator(
            base_data_path=str(self.base_data_path),
            baseline_period=(1980, 2010)
        )
        
        # 1. No cache - calculate baseline each time
        start = time.time()
        for _ in range(3):
            thresholds = calculator.calculate_baseline_percentiles(county_info['bounds'])
        results['no_cache'] = time.time() - start
        
        # 2. With cache
        cache = {}
        start = time.time()
        for _ in range(3):
            cache_key = f"{county_info['geoid']}_baseline"
            if cache_key in cache:
                thresholds = cache[cache_key]
            else:
                thresholds = calculator.calculate_baseline_percentiles(county_info['bounds'])
                cache[cache_key] = thresholds
        results['with_cache'] = time.time() - start
        
        # 3. Pre-computed and saved to disk
        cache_file = Path(f"baseline_cache_{county_info['geoid']}.pkl")
        
        # First save
        start_save = time.time()
        with open(cache_file, 'wb') as f:
            pickle.dump(thresholds, f)
        save_time = time.time() - start_save
        
        # Then load multiple times
        start = time.time()
        for _ in range(3):
            with open(cache_file, 'rb') as f:
                thresholds = pickle.load(f)
        results['disk_cache'] = time.time() - start + save_time
        
        # Cleanup
        cache_file.unlink()
        
        print("\nCaching Results:")
        for strategy, timing in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {strategy}: {timing:.3f}s")
            speedup = results['no_cache'] / timing
            print(f"    Speedup: {speedup:.1f}x")
            
        return results
    
    def estimate_full_scale_performance(self, sample_size: int = 20) -> Dict:
        """
        Estimate performance for processing all counties.
        """
        print(f"\nEstimating full-scale performance...")
        
        # Get sample counties
        processor = ParallelClimateProcessor(
            counties_shapefile_path=str(self.shapefile_path),
            base_data_path=str(self.base_data_path)
        )
        
        total_counties = len(processor.counties)
        sample_counties = processor.counties.sample(n=min(sample_size, total_counties))
        
        # Time sample processing
        start = time.time()
        df = processor.process_parallel(
            counties_subset=sample_counties,
            scenarios=['historical', 'ssp245'],
            historical_period=(1980, 2010),
            future_period=(2040, 2070),
            n_workers=min(mp.cpu_count() - 1, 16)
        )
        sample_time = time.time() - start
        
        # Calculate estimates
        time_per_county = sample_time / len(sample_counties)
        estimated_total_time = time_per_county * total_counties
        
        # Estimate with different worker counts
        estimates = {
            'sample_counties': len(sample_counties),
            'sample_time_seconds': sample_time,
            'time_per_county': time_per_county,
            'total_counties': total_counties,
            'estimated_hours': estimated_total_time / 3600
        }
        
        # Estimate for different worker counts
        for workers in [4, 8, 16, 32, 64]:
            # Assume linear speedup up to diminishing returns
            efficiency = min(1.0, workers / 16) * 0.8  # 80% efficiency at best
            estimated_time = estimated_total_time / (workers * efficiency)
            estimates[f'hours_with_{workers}_workers'] = estimated_time / 3600
        
        print(f"\nFull Scale Estimates:")
        print(f"  Total counties: {total_counties:,}")
        print(f"  Sample size: {len(sample_counties)}")
        print(f"  Time per county: {time_per_county:.1f}s")
        print(f"\nEstimated processing time:")
        for workers in [4, 8, 16, 32, 64]:
            if f'hours_with_{workers}_workers' in estimates:
                hours = estimates[f'hours_with_{workers}_workers']
                print(f"  {workers} workers: {hours:.1f} hours ({hours/24:.1f} days)")
                
        return estimates
    
    def generate_optimization_report(self) -> None:
        """
        Generate a comprehensive optimization report.
        """
        print("\n" + "="*60)
        print("PERFORMANCE OPTIMIZATION REPORT")
        print("="*60)
        
        # Test with a sample county
        processor = ParallelClimateProcessor(
            counties_shapefile_path=str(self.shapefile_path),
            base_data_path=str(self.base_data_path)
        )
        
        sample_county = processor.counties.iloc[0]
        county_info = processor.prepare_county_info(sample_county)
        
        # Run all tests
        self.results['single_county_profile'] = self.profile_single_county(county_info)
        self.results['io_strategies'] = self.test_io_strategies(county_info)
        self.results['parallelization'] = self.test_parallelization_strategies(20)
        self.results['caching'] = self.test_caching_strategies(county_info)
        self.results['full_scale_estimate'] = self.estimate_full_scale_performance(20)
        
        # Save results
        with open('performance_report.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\nReport saved to performance_report.json")
        
        # Key recommendations
        print("\n" + "="*60)
        print("KEY RECOMMENDATIONS")
        print("="*60)
        
        print("\n1. I/O Optimization:")
        io_results = self.results['io_strategies']
        best_io = min(io_results.items(), key=lambda x: x[1])[0]
        print(f"   - Use '{best_io}' strategy for best I/O performance")
        
        print("\n2. Parallelization:")
        para_results = self.results['parallelization']
        worker_results = {k: v for k, v in para_results.items() if k.startswith('workers_')}
        best_workers = max(worker_results.items(), 
                          key=lambda x: x[1]['counties_per_second'])[0]
        print(f"   - Optimal worker count: {best_workers}")
        
        print("\n3. Caching:")
        cache_results = self.results['caching']
        print(f"   - Implement baseline caching for {cache_results['no_cache']/cache_results['with_cache']:.1f}x speedup")
        
        print("\n4. Full Scale Processing:")
        estimates = self.results['full_scale_estimate']
        print(f"   - Processing all {estimates['total_counties']:,} counties")
        print(f"   - Recommended: 16-32 workers")
        print(f"   - Estimated time: {estimates.get('hours_with_16_workers', 'N/A'):.1f} hours")


def create_optimization_plan(base_data_path: str, shapefile_path: str) -> Dict:
    """
    Create a specific optimization plan based on performance analysis.
    """
    recommendations = {
        'immediate_optimizations': [
            {
                'name': 'Implement baseline caching',
                'impact': 'High',
                'effort': 'Low',
                'description': 'Cache baseline percentiles to avoid recalculation',
                'estimated_speedup': '3-5x for repeated county processing'
            },
            {
                'name': 'Optimize NetCDF I/O',
                'impact': 'High', 
                'effort': 'Medium',
                'description': 'Pre-select spatial regions before loading full datasets',
                'estimated_speedup': '2-3x for data loading'
            },
            {
                'name': 'Adjust worker count',
                'impact': 'Medium',
                'effort': 'Low',
                'description': 'Use optimal number of workers based on CPU count',
                'estimated_speedup': 'Linear up to CPU count'
            }
        ],
        'future_optimizations': [
            {
                'name': 'Implement spatial indexing',
                'impact': 'High',
                'effort': 'High',
                'description': 'Pre-process data into spatially indexed chunks',
                'estimated_speedup': '5-10x for data access'
            },
            {
                'name': 'Use Zarr format',
                'impact': 'High',
                'effort': 'High',
                'description': 'Convert NetCDF to Zarr for cloud-optimized access',
                'estimated_speedup': '3-5x for parallel reads'
            },
            {
                'name': 'GPU acceleration',
                'impact': 'Very High',
                'effort': 'Very High',
                'description': 'Use GPU for percentile calculations',
                'estimated_speedup': '10-50x for computation'
            }
        ],
        'scaling_strategy': {
            'small_scale': '< 100 counties: Use 4-8 workers, simple parallelization',
            'medium_scale': '100-1000 counties: Implement caching, optimize I/O, 16 workers',
            'large_scale': '> 1000 counties: Distributed processing, cloud storage, 32+ workers',
            'full_scale': 'All 3200+ counties: Consider HPC cluster or cloud compute'
        }
    }
    
    return recommendations


# Example usage
if __name__ == "__main__":
    evaluator = PerformanceEvaluator(
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp"
    )
    
    # Run comprehensive evaluation
    evaluator.generate_optimization_report()
    
    # Create optimization plan
    plan = create_optimization_plan(
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp"
    )
    
    print("\n" + "="*60)
    print("OPTIMIZATION PLAN")
    print("="*60)
    
    for category, items in plan.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        if isinstance(items, list):
            for item in items:
                print(f"\n  {item['name']}")
                print(f"    Impact: {item['impact']}, Effort: {item['effort']}")
                print(f"    {item['description']}")
                print(f"    Expected speedup: {item['estimated_speedup']}")
        elif isinstance(items, dict):
            for k, v in items.items():
                print(f"  {k}: {v}")