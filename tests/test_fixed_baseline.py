#!/usr/bin/env python3
"""
Test the fixed baseline percentile calculation
"""

import pandas as pd
import sys
sys.path.append('../src')
from parallel_xclim_processor_fixed import ParallelXclimProcessorFixed

def test_fixed_baseline():
    """
    Test with fixed baseline period for percentiles
    """
    print("TESTING FIXED BASELINE PERCENTILE CALCULATION")
    print("="*60)
    
    # Initialize with 30-year baseline
    processor = ParallelXclimProcessorFixed(
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010)  # 30-year climatological baseline
    )
    
    # Test with just 2 counties
    test_geoids = ['31039', '53069']  # Cuming NE, Wahkiakum WA
    processor.counties = processor.counties[processor.counties['GEOID'].isin(test_geoids)].copy()
    
    print(f"\nTesting with {len(processor.counties)} counties:")
    for idx, county in processor.counties.iterrows():
        print(f"  - {county['NAME']}, {county['STATEFP']}")
    
    print("\nProcessing with fixed baseline...")
    print("  Baseline: 1980-2010 (30 years)")
    print("  Historical analysis: 2005-2010")
    print("  Future projection: 2040-2045")
    
    try:
        # Process with fixed baseline
        df = processor.process_xclim_parallel(
            scenarios=['historical', 'ssp245'],
            variables=['tas', 'tasmax', 'tasmin', 'pr'],
            historical_period=(2005, 2010),  # Recent historical
            future_period=(2040, 2045),      # Mid-century
            n_chunks=1  # Single chunk for 2 counties
        )
        
        # Save results
        output_file = "../results/test_fixed_baseline_results.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Total records: {len(df)}")
        
        # Analyze percentile values
        print("\n" + "-"*60)
        print("PERCENTILE INDICATOR ANALYSIS")
        print("-"*60)
        
        for county_name in df['NAME'].unique():
            county_data = df[df['NAME'] == county_name]
            
            print(f"\n{county_name}:")
            
            # Historical percentiles
            hist_data = county_data[county_data['scenario'] == 'historical']
            if 'tx90p_percent' in hist_data.columns:
                tx90p_hist = hist_data['tx90p_percent'].mean()
                tn10p_hist = hist_data['tn10p_percent'].mean()
                print(f"  Historical (2005-2010):")
                print(f"    tx90p: {tx90p_hist:.1f}% (should be ~10% if similar to baseline)")
                print(f"    tn10p: {tn10p_hist:.1f}% (should be ~10% if similar to baseline)")
            
            # Future percentiles
            fut_data = county_data[county_data['scenario'] == 'ssp245']
            if not fut_data.empty and 'tx90p_percent' in fut_data.columns:
                tx90p_fut = fut_data['tx90p_percent'].mean()
                tn10p_fut = fut_data['tn10p_percent'].mean()
                print(f"  Future SSP2-4.5 (2040-2045):")
                print(f"    tx90p: {tx90p_fut:.1f}% (>10% indicates more hot extremes)")
                print(f"    tn10p: {tn10p_fut:.1f}% (<10% indicates fewer cold extremes)")
                
                # Calculate changes
                if tx90p_hist > 0:
                    print(f"  Changes from recent historical:")
                    print(f"    tx90p change: {tx90p_fut - tx90p_hist:+.1f} percentage points")
                    print(f"    tn10p change: {tn10p_fut - tn10p_hist:+.1f} percentage points")
        
        # Show temperature changes
        print("\n" + "-"*60)
        print("TEMPERATURE CHANGES")
        print("-"*60)
        
        summary = df.groupby(['NAME', 'scenario']).agg({
            'tg_mean_C': 'mean',
            'tx_days_above_90F': 'mean',
            'tn_days_below_32F': 'mean'
        }).round(1)
        
        print("\nMean values by period:")
        print(summary.to_string())
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_baseline()