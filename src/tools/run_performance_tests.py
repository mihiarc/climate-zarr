#!/usr/bin/env python3
"""
Run performance tests and generate CSV results.

This script tests different optimization strategies and saves results to:
/home/mihiarc/repos/claude_climate/results/performance_test_results.csv
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
import psutil
import json

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.unified_calculator import UnifiedClimateCalculator
from src.core.unified_processor import UnifiedParallelProcessor
from src.utils.file_operations import ensure_directory


def test_single_county_performance(
    calculator_type: str,
    county_info: dict,
    scenarios: list,
    indicators_config: dict,
    **kwargs
) -> dict:
    """Test performance for a single county with specified calculator."""
    
    # Create calculator based on type
    if calculator_type == "baseline":
        # For baseline comparison, use unified calculator without merged baselines
        calculator = UnifiedClimateCalculator(
            base_data_path=kwargs['base_data_path'],
            merged_baseline_path=None,
            baseline_period=(1980, 2010)
        )
    elif calculator_type == "unified":
        calculator = UnifiedClimateCalculator(
            base_data_path=kwargs['base_data_path'],
            merged_baseline_path=kwargs.get('merged_baseline_path'),
            baseline_period=(1980, 2010)
        )
    elif calculator_type == "unified_no_cache":
        calculator = UnifiedClimateCalculator(
            base_data_path=kwargs['base_data_path'],
            merged_baseline_path=None,  # No merged baselines
            baseline_period=(1980, 2010)
        )
    else:
        raise ValueError(f"Unknown calculator type: {calculator_type}")
    
    # Time the calculation
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        results = calculator.calculate_indicators(
            scenarios=scenarios,
            county_bounds=county_info['bounds'],
            county_info={
                'geoid': county_info['GEOID'],
                'name': county_info['NAME'],
                'state': county_info.get('STATE_NAME', 'Unknown')
            },
            indicators_config=indicators_config
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'success': True,
            'time_seconds': end_time - start_time,
            'memory_mb': end_memory - start_memory,
            'n_records': len(results),
            'error': None
        }
        
    except Exception as e:
        end_time = time.time()
        return {
            'success': False,
            'time_seconds': end_time - start_time,
            'memory_mb': 0,
            'n_records': 0,
            'error': str(e)
        }


def test_parallel_performance(
    processor_type: str,
    counties_gdf,
    n_counties: int,
    n_workers: int,
    scenarios: list,
    indicators_config: dict,
    **kwargs
) -> dict:
    """Test parallel processing performance."""
    
    # Select test counties
    test_counties = counties_gdf.sample(n=min(n_counties, len(counties_gdf)))
    
    # Create processor based on type
    if processor_type == "baseline":
        from src.core.parallel_processor import ParallelClimateProcessor
        processor = ParallelClimateProcessor(
            counties_shapefile_path=kwargs['shapefile_path'],
            base_data_path=kwargs['base_data_path']
        )
    elif processor_type == "unified":
        processor = UnifiedParallelProcessor(
            shapefile_path=kwargs['shapefile_path'],
            base_data_path=kwargs['base_data_path'],
            merged_baseline_path=kwargs.get('merged_baseline_path'),
            n_workers=n_workers
        )
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")
    
    # Time the processing
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        if processor_type == "baseline":
            # Use old API
            results_df = processor.process_parallel(
                counties_subset=test_counties,
                scenarios=scenarios,
                historical_period=(2009, 2010),
                future_period=(2040, 2041),
                n_workers=n_workers,
                variables=['tas', 'tasmax', 'tasmin', 'pr']
            )
        else:
            # Use new API
            results_df = processor.process_all_counties(
                scenarios=scenarios,
                indicators_config=indicators_config,
                counties_filter=test_counties['GEOID'].tolist(),
                batch_size=max(1, n_counties // (n_workers * 4))
            )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        elapsed = end_time - start_time
        
        return {
            'success': True,
            'time_seconds': elapsed,
            'counties_per_second': n_counties / elapsed,
            'memory_mb': end_memory - start_memory,
            'n_records': len(results_df),
            'error': None
        }
        
    except Exception as e:
        end_time = time.time()
        return {
            'success': False,
            'time_seconds': end_time - start_time,
            'counties_per_second': 0,
            'memory_mb': 0,
            'n_records': 0,
            'error': str(e)
        }


def run_all_tests(base_data_path: str, shapefile_path: str, merged_baseline_path: str = None):
    """Run comprehensive performance tests and save results to CSV."""
    
    print("Starting performance tests...")
    
    # Ensure results directory exists
    results_dir = Path("/home/mihiarc/repos/claude_climate/results")
    ensure_directory(results_dir)
    
    # Load counties
    import geopandas as gpd
    counties_gdf = gpd.read_file(shapefile_path)
    counties_gdf['bounds'] = counties_gdf.geometry.bounds.values.tolist()
    
    # Test configuration
    scenarios = ['historical']
    indicators_config = {
        'tx90p': {'xclim_func': 'tx90p', 'variable': 'tasmax', 'freq': 'YS'},
        'tn10p': {'xclim_func': 'tn10p', 'variable': 'tasmin', 'freq': 'YS'},
        'tx_days_above_90F': {'xclim_func': 'tx_days_above', 'variable': 'tasmax', 
                              'thresh': '305.37 K', 'freq': 'YS'},
        'tn_days_below_32F': {'xclim_func': 'tn_days_below', 'variable': 'tasmin',
                              'thresh': '273.15 K', 'freq': 'YS'},
        'tg_mean': {'xclim_func': 'tg_mean', 'variable': 'tas', 'freq': 'YS'},
        'precip_accumulation': {'xclim_func': 'precip_accumulation', 'variable': 'pr', 'freq': 'YS'},
        'days_precip_over_25.4mm': {'xclim_func': 'wetdays', 'variable': 'pr',
                                    'thresh': '0.000294 kg m-2 s-1', 'freq': 'YS'}
    }
    
    # Results storage
    all_results = []
    
    # Test 1: Single county performance comparison
    print("\n1. Testing single county performance...")
    test_county = counties_gdf.iloc[0].to_dict()
    
    # Test configurations with descriptive names
    test_configs = [
        ("baseline", "Baseline Implementation"),
        ("unified_no_cache", "Unified (No Pre-merged Baselines)"),
        ("unified", "Unified (With Pre-merged Baselines)")
    ]
    
    for calc_type, description in test_configs:
        if calc_type == "baseline" and not Path("src/core/climate_indicator_calculator.py").exists():
            print(f"  Skipping {calc_type} - file not found")
            continue
            
        print(f"  Testing {description}...")
        
        result = test_single_county_performance(
            calculator_type=calc_type,
            county_info=test_county,
            scenarios=scenarios,
            indicators_config=indicators_config,
            base_data_path=base_data_path,
            merged_baseline_path=merged_baseline_path if calc_type == "unified" else None
        )
        
        all_results.append({
            'test_type': 'single_county',
            'implementation': calc_type,
            'n_counties': 1,
            'n_workers': 1,
            'time_seconds': result['time_seconds'],
            'counties_per_second': 1 / result['time_seconds'] if result['success'] else 0,
            'memory_mb': result['memory_mb'],
            'success': result['success'],
            'error': result['error'],
            'timestamp': datetime.now().isoformat()
        })
    
    # Test 2: Parallel processing with different worker counts
    print("\n2. Testing parallel processing scalability...")
    
    for n_workers in [1, 2, 4, 8, min(16, mp.cpu_count())]:
        for proc_type in ["baseline", "unified"]:
            if proc_type == "baseline" and not Path("src/core/parallel_processor.py").exists():
                print(f"  Skipping {proc_type} - file not found")
                continue
                
            print(f"  Testing {proc_type} processor with {n_workers} workers...")
            
            result = test_parallel_performance(
                processor_type=proc_type,
                counties_gdf=counties_gdf,
                n_counties=20,  # Test with 20 counties
                n_workers=n_workers,
                scenarios=scenarios,
                indicators_config=indicators_config,
                base_data_path=base_data_path,
                shapefile_path=shapefile_path,
                merged_baseline_path=merged_baseline_path if proc_type == "unified" else None
            )
            
            all_results.append({
                'test_type': 'parallel_processing',
                'implementation': proc_type,
                'n_counties': 20,
                'n_workers': n_workers,
                'time_seconds': result['time_seconds'],
                'counties_per_second': result['counties_per_second'],
                'memory_mb': result['memory_mb'],
                'success': result['success'],
                'error': result['error'],
                'timestamp': datetime.now().isoformat()
            })
    
    # Test 3: Batch size impact
    print("\n3. Testing batch size impact...")
    
    optimal_workers = min(8, mp.cpu_count() // 2)
    for batch_factor in [1, 2, 4, 8]:
        n_counties = 40
        batch_size = max(1, n_counties // (optimal_workers * batch_factor))
        
        print(f"  Testing batch size {batch_size} (factor {batch_factor})...")
        
        processor = UnifiedParallelProcessor(
            shapefile_path=shapefile_path,
            base_data_path=base_data_path,
            merged_baseline_path=merged_baseline_path,
            n_workers=optimal_workers
        )
        
        test_counties = counties_gdf.sample(n=n_counties)
        
        start_time = time.time()
        try:
            results_df = processor.process_all_counties(
                scenarios=scenarios,
                indicators_config=indicators_config,
                counties_filter=test_counties['GEOID'].tolist(),
                batch_size=batch_size
            )
            elapsed = time.time() - start_time
            success = True
            error = None
        except Exception as e:
            elapsed = time.time() - start_time
            success = False
            error = str(e)
            results_df = pd.DataFrame()
        
        all_results.append({
            'test_type': 'batch_size_test',
            'implementation': f'unified_batch_{batch_size}',
            'n_counties': n_counties,
            'n_workers': optimal_workers,
            'time_seconds': elapsed,
            'counties_per_second': n_counties / elapsed if success else 0,
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    
    # Calculate speedup factors
    baseline_single = results_df[
        (results_df['test_type'] == 'single_county') & 
        (results_df['implementation'] == 'baseline')
    ]['time_seconds'].values
    
    if len(baseline_single) > 0:
        baseline_time = baseline_single[0]
        results_df['speedup_vs_baseline'] = results_df.apply(
            lambda row: baseline_time / row['time_seconds'] if row['test_type'] == 'single_county' else np.nan,
            axis=1
        )
    
    # Save results
    output_path = results_dir / "performance_test_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE TEST SUMMARY")
    print("="*60)
    
    # Single county results
    single_county_results = results_df[results_df['test_type'] == 'single_county']
    if not single_county_results.empty:
        print("\nSingle County Performance:")
        baseline_time = single_county_results[single_county_results['implementation'] == 'baseline']['time_seconds'].values
        baseline_time = baseline_time[0] if len(baseline_time) > 0 else None
        
        for _, row in single_county_results.iterrows():
            if baseline_time:
                speedup = baseline_time / row['time_seconds']
                print(f"  {row['implementation']}: {row['time_seconds']:.2f}s (speedup: {speedup:.1f}x)")
            else:
                print(f"  {row['implementation']}: {row['time_seconds']:.2f}s")
        
        # Show specific comparison for merged baselines
        unified_no_cache = single_county_results[single_county_results['implementation'] == 'unified_no_cache']['time_seconds'].values
        unified_with_cache = single_county_results[single_county_results['implementation'] == 'unified']['time_seconds'].values
        
        if len(unified_no_cache) > 0 and len(unified_with_cache) > 0:
            cache_speedup = unified_no_cache[0] / unified_with_cache[0]
            print(f"\n  Pre-merged baseline speedup: {cache_speedup:.1f}x")
    
    # Parallel scaling results
    parallel_results = results_df[
        (results_df['test_type'] == 'parallel_processing') &
        (results_df['implementation'] == 'unified')
    ]
    if not parallel_results.empty:
        print("\nParallel Processing Scaling:")
        for _, row in parallel_results.iterrows():
            print(f"  {row['n_workers']} workers: {row['counties_per_second']:.2f} counties/second")
    
    # Best configuration
    best_config = results_df.loc[results_df['counties_per_second'].idxmax()]
    print(f"\nBest Configuration:")
    print(f"  Type: {best_config['implementation']}")
    print(f"  Workers: {best_config['n_workers']}")
    print(f"  Performance: {best_config['counties_per_second']:.2f} counties/second")
    
    # Save summary as JSON
    summary = {
        'test_date': datetime.now().isoformat(),
        'total_tests': len(results_df),
        'successful_tests': len(results_df[results_df['success']]),
        'best_single_county_time': single_county_results['time_seconds'].min() if not single_county_results.empty else None,
        'best_parallel_rate': results_df['counties_per_second'].max(),
        'optimal_workers': int(best_config['n_workers']),
        'system_info': {
            'cpu_count': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
    }
    
    with open(results_dir / "performance_test_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results_df


if __name__ == "__main__":
    # Configuration
    BASE_DATA_PATH = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    SHAPEFILE_PATH = "/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp"
    MERGED_BASELINE_PATH = "/media/mihiarc/RPA1TB/CLIMATE_DATA/merged_baselines/merged_baselines.pkl"
    
    # Check if paths exist
    if not Path(BASE_DATA_PATH).exists():
        print(f"Error: Base data path not found: {BASE_DATA_PATH}")
        print("Please update the path to your climate data location.")
        sys.exit(1)
    
    if not Path(SHAPEFILE_PATH).exists():
        print(f"Error: Shapefile not found: {SHAPEFILE_PATH}")
        print("Please update the path to your counties shapefile.")
        sys.exit(1)
    
    # Run tests
    results_df = run_all_tests(
        base_data_path=BASE_DATA_PATH,
        shapefile_path=SHAPEFILE_PATH,
        merged_baseline_path=MERGED_BASELINE_PATH if Path(MERGED_BASELINE_PATH).exists() else None
    )