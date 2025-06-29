#!/usr/bin/env python3
"""
Run NEX-GDDP climate analysis using both spatial and temporal chunking strategies
This script intelligently selects the best parallelization strategy based on the data characteristics
"""

import multiprocessing as mp
from parallel_nex_gddp_processor import ParallelNEXGDDP_CountyProcessor
import pandas as pd
import time
import sys
import numpy as np

def estimate_data_size(variable, scenario, time_range_years):
    """
    Estimate dataset size to choose optimal strategy
    """
    # Rough estimates based on NEX-GDDP data characteristics
    n_counties = 3235
    n_days = time_range_years * 365
    n_lat = 600
    n_lon = 1440
    bytes_per_value = 4  # float32
    
    # Full dataset size in GB
    full_size_gb = (n_days * n_lat * n_lon * bytes_per_value) / (1024**3)
    
    return {
        'n_counties': n_counties,
        'n_days': n_days,
        'time_range_years': time_range_years,
        'estimated_size_gb': full_size_gb
    }

def choose_strategy(variable, scenario, time_range_years):
    """
    Choose optimal parallelization strategy based on data characteristics
    """
    data_info = estimate_data_size(variable, scenario, time_range_years)
    
    # Decision logic
    if time_range_years > 20:
        # Long time series - use temporal chunks
        strategy = 'temporal'
        params = {
            'years_per_chunk': 5 if time_range_years > 30 else 10
        }
        reason = f"Long time series ({time_range_years} years) - temporal chunking is optimal"
    
    elif data_info['estimated_size_gb'] < 10:
        # Small dataset - load to memory
        strategy = 'memory'
        params = {}
        reason = f"Small dataset (~{data_info['estimated_size_gb']:.1f} GB) - fits in memory"
    
    else:
        # Default to spatial chunks
        strategy = 'spatial'
        params = {
            'n_chunks': min(4, mp.cpu_count())
        }
        reason = f"Standard dataset - spatial chunking with {params['n_chunks']} chunks"
    
    return strategy, params, reason

def process_with_optimal_strategy(processor, variable, scenario, start_year, end_year):
    """
    Process data using the optimal strategy for the given parameters
    """
    time_range_years = end_year - start_year + 1
    strategy, params, reason = choose_strategy(variable, scenario, time_range_years)
    
    print(f"\nStrategy selection:")
    print(f"  - Variable: {variable}")
    print(f"  - Scenario: {scenario}")
    print(f"  - Time range: {start_year}-{end_year} ({time_range_years} years)")
    print(f"  - Selected: {strategy}")
    print(f"  - Reason: {reason}")
    
    # Map strategy names to process_parallel strategy parameter
    strategy_map = {
        'spatial': 'spatial_chunks',
        'temporal': 'temporal_chunks',
        'memory': 'memory'
    }
    
    # Execute with selected strategy
    start_time = time.time()
    
    results = processor.process_parallel(
        variable=variable,
        scenario=scenario,
        strategy=strategy_map[strategy],
        start_year=start_year,
        end_year=end_year,
        **params
    )
    
    execution_time = time.time() - start_time
    print(f"  - Execution time: {execution_time:.1f} seconds")
    
    return results, strategy, execution_time

def run_hybrid_parallel_processing():
    """
    Main function to run climate analysis with hybrid parallelization
    """
    total_start_time = time.time()
    
    # Initialize processor
    print("Initializing processor...")
    processor = ParallelNEXGDDP_CountyProcessor(
        counties_shapefile_path="/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    )
    
    print(f"System has {mp.cpu_count()} CPU cores available")
    
    # Define analysis configurations
    analyses = [
        # Historical analyses (longer time series)
        {
            'variable': 'tas',
            'scenario': 'historical',
            'start_year': 1950,
            'end_year': 2014,
            'description': 'Historical temperature (65 years)'
        },
        {
            'variable': 'pr',
            'scenario': 'historical', 
            'start_year': 1980,
            'end_year': 2010,
            'description': 'Historical precipitation baseline (31 years)'
        },
        
        # Future projections (moderate time series)
        {
            'variable': 'tas',
            'scenario': 'ssp245',
            'start_year': 2040,
            'end_year': 2070,
            'description': 'Mid-century temperature SSP2-4.5 (31 years)'
        },
        {
            'variable': 'tas',
            'scenario': 'ssp585',
            'start_year': 2040,
            'end_year': 2070,
            'description': 'Mid-century temperature SSP5-8.5 (31 years)'
        },
        
        # Near-term projections (shorter time series)
        {
            'variable': 'tasmax',
            'scenario': 'ssp245',
            'start_year': 2020,
            'end_year': 2030,
            'description': 'Near-term maximum temperature (11 years)'
        },
        {
            'variable': 'pr',
            'scenario': 'ssp585',
            'start_year': 2070,
            'end_year': 2100,
            'description': 'End-century precipitation SSP5-8.5 (31 years)'
        }
    ]
    
    # Process each analysis
    all_results = {}
    strategy_stats = {'spatial_chunks': 0, 'temporal_chunks': 0, 'memory': 0}
    execution_times = []
    
    print("\n" + "="*80)
    print("STARTING HYBRID PARALLEL PROCESSING")
    print("="*80)
    
    for i, analysis in enumerate(analyses, 1):
        print(f"\n[{i}/{len(analyses)}] {analysis['description']}")
        print("-" * 60)
        
        try:
            results, strategy, exec_time = process_with_optimal_strategy(
                processor,
                analysis['variable'],
                analysis['scenario'],
                analysis['start_year'],
                analysis['end_year']
            )
            
            # Store results
            key = f"{analysis['variable']}_{analysis['scenario']}_{analysis['start_year']}_{analysis['end_year']}"
            all_results[key] = {
                'results': results,
                'metadata': analysis,
                'strategy': strategy,
                'execution_time': exec_time
            }
            
            # Update statistics
            strategy_map = {
                'spatial': 'spatial_chunks',
                'temporal': 'temporal_chunks',
                'memory': 'memory'
            }
            strategy_stats[strategy_map[strategy]] += 1
            execution_times.append(exec_time)
            
            print(f"✓ Successfully completed")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    # Save all results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    for key, data in all_results.items():
        output_file = f"hybrid_{key}"
        
        # Create dataframe from results
        df_data = []
        for geoid, county_data in data['results'].items():
            row = {
                'GEOID': geoid,
                'name': county_data['name'],
                'state': county_data['state'],
                'variable': county_data['variable'],
                'scenario': data['metadata']['scenario'],
                'start_year': data['metadata']['start_year'],
                'end_year': data['metadata']['end_year'],
                'mean_value': np.mean(county_data['values']),
                'std_value': np.std(county_data['values']),
                'min_value': np.min(county_data['values']),
                'max_value': np.max(county_data['values']),
                'strategy_used': data['strategy'],
                'execution_time': data['execution_time']
            }
            
            # Add variable-specific metrics
            if county_data['variable'] == 'pr':
                # Convert to mm/year
                annual_values = county_data['values'] * 86400 * 365.25
                row['annual_mean_mm'] = np.mean(annual_values)
            else:
                # Convert to Celsius
                celsius_values = county_data['values'] - 273.15
                row['mean_celsius'] = np.mean(celsius_values)
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(f"{output_file}.csv", index=False)
        print(f"Saved: {output_file}.csv ({len(df)} counties)")
    
    # Print summary statistics
    total_time = time.time() - total_start_time
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*80)
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")
    print(f"\nStrategy usage:")
    for strategy, count in strategy_stats.items():
        if count > 0:
            print(f"  - {strategy}: {count} times")
    
    if execution_times:
        print(f"\nExecution time statistics:")
        print(f"  - Mean: {np.mean(execution_times):.1f} seconds")
        print(f"  - Min: {np.min(execution_times):.1f} seconds")
        print(f"  - Max: {np.max(execution_times):.1f} seconds")
    
    print(f"\nTotal analyses completed: {len(all_results)}/{len(analyses)}")
    print(f"Counties processed per analysis: {len(processor.counties)}")
    
    # Create summary report
    summary_file = "hybrid_processing_summary.json"
    import json
    summary = {
        'total_execution_time_minutes': total_time/60,
        'analyses_completed': len(all_results),
        'total_analyses': len(analyses),
        'counties_per_analysis': len(processor.counties),
        'strategy_usage': strategy_stats,
        'execution_times': {
            'mean_seconds': np.mean(execution_times) if execution_times else 0,
            'min_seconds': np.min(execution_times) if execution_times else 0,
            'max_seconds': np.max(execution_times) if execution_times else 0
        },
        'analyses_details': [
            {
                'description': data['metadata']['description'],
                'variable': data['metadata']['variable'],
                'scenario': data['metadata']['scenario'],
                'years': f"{data['metadata']['start_year']}-{data['metadata']['end_year']}",
                'strategy': data['strategy'],
                'execution_time_seconds': data['execution_time']
            }
            for data in all_results.values()
        ]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed summary saved to: {summary_file}")

if __name__ == "__main__":
    run_hybrid_parallel_processing()