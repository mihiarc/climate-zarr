#!/usr/bin/env python3
"""
Run NEX-GDDP climate analysis using spatial chunks parallelization with 16 CPUs
"""

import multiprocessing as mp
from parallel_nex_gddp_processor import ParallelNEXGDDP_CountyProcessor
import pandas as pd
import time
import sys

def main():
    # Start timer
    start_time = time.time()
    
    # Initialize processor
    print("Initializing processor...")
    processor = ParallelNEXGDDP_CountyProcessor(
        counties_shapefile_path="/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    )
    
    # Variables and scenarios to process
    variables = ['tas', 'tasmin', 'tasmax', 'pr']
    scenarios = ['historical', 'ssp245', 'ssp585']
    
    # Time periods
    historical_period = (1980, 2010)
    future_period = (2040, 2070)
    
    # Process each variable
    all_results = {}
    
    for variable in variables:
        print(f"\n{'='*60}")
        print(f"Processing variable: {variable}")
        print(f"{'='*60}")
        
        variable_results = {}
        
        for scenario in scenarios:
            print(f"\nProcessing {variable} - {scenario}...")
            
            try:
                # Determine time period
                if scenario == 'historical':
                    start_year, end_year = historical_period
                else:
                    start_year, end_year = future_period
                
                # Process with spatial chunks using 16 CPUs
                results = processor.process_parallel(
                    variable=variable,
                    scenario=scenario,
                    strategy='spatial_chunks',
                    n_chunks=16,  # Use 16 CPU cores
                    start_year=start_year,
                    end_year=end_year
                )
                
                variable_results[scenario] = results
                print(f"✓ Completed {variable} - {scenario}")
                
            except Exception as e:
                print(f"✗ Error processing {variable} - {scenario}: {e}")
                continue
        
        # Create comparison dataframe for this variable
        if variable_results:
            print(f"\nCreating comparison dataframe for {variable}...")
            df = processor.create_comparison_dataframe(variable_results, variable)
            
            # Save results
            output_file = f"parallel_16cpu_{variable}_analysis"
            processor.save_results(df, output_file)
            
            all_results[variable] = df
            
            # Print summary statistics
            print(f"\nSummary for {variable}:")
            print(f"- Counties processed: {len(df)}")
            print(f"- Scenarios included: {list(variable_results.keys())}")
            
            if 'historical' in variable_results and 'ssp245' in variable_results:
                if variable == 'pr':
                    change_col = 'change_ssp245_percent'
                    if change_col in df.columns:
                        print(f"- Mean precipitation change (SSP2-4.5): {df[change_col].mean():.1f}%")
                else:
                    change_col = 'change_ssp245_C'
                    if change_col in df.columns:
                        print(f"- Mean temperature change (SSP2-4.5): {df[change_col].mean():.2f}°C")
    
    # Create combined summary report
    if all_results:
        print("\n" + "="*60)
        print("CREATING COMBINED SUMMARY REPORT")
        print("="*60)
        
        summary_rows = []
        for variable, df in all_results.items():
            for idx, row in df.iterrows():
                summary_row = {
                    'GEOID': row['GEOID'],
                    'NAME': row['NAME'],
                    'STATE': row['STATE'],
                    'variable': variable
                }
                
                # Add variable-specific metrics
                for col in df.columns:
                    if col not in ['GEOID', 'NAME', 'STATE']:
                        summary_row[f"{variable}_{col}"] = row[col]
                
                summary_rows.append(summary_row)
        
        # Create wide-format summary
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.pivot_table(
            index=['GEOID', 'NAME', 'STATE'],
            columns='variable',
            values=[col for col in summary_df.columns if col not in ['GEOID', 'NAME', 'STATE', 'variable']],
            aggfunc='first'
        )
        
        # Save combined summary
        summary_df.to_csv('parallel_16cpu_all_variables_summary.csv')
        print("✓ Saved combined summary to parallel_16cpu_all_variables_summary.csv")
    
    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total execution time: {execution_time/60:.1f} minutes")
    print(f"Average time per variable: {execution_time/len(variables)/60:.1f} minutes")
    print(f"Counties processed: {len(processor.counties)}")
    print(f"CPU cores used: 16")
    print(f"Parallelization strategy: spatial_chunks")

if __name__ == "__main__":
    # Check system CPU count
    available_cpus = mp.cpu_count()
    print(f"System has {available_cpus} CPU cores available")
    
    if available_cpus < 16:
        print(f"WARNING: Requesting 16 cores but only {available_cpus} available.")
        print("The script will still run but may not achieve expected performance.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    main()