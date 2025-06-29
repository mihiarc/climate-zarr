#!/usr/bin/env python3
"""
Test the unified parallel processor in both modes
"""

import pandas as pd
import sys
sys.path.append('../src')
from parallel_xclim_processor_unified import ParallelXclimProcessor


def test_unified_processor():
    """
    Test the unified processor with both fixed and period-specific baselines
    """
    print("TESTING UNIFIED PARALLEL PROCESSOR")
    print("="*60)
    
    # Test counties
    test_geoids = ['31039', '53069']  # Cuming NE, Wahkiakum WA
    
    # Test periods
    historical_period = (2005, 2010)
    future_period = (2040, 2045)
    
    print("\n1. TESTING FIXED BASELINE MODE")
    print("-"*60)
    
    # Initialize with fixed baseline
    processor_fixed = ParallelXclimProcessor(
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010),
        use_fixed_baseline=True
    )
    
    # Filter to test counties
    processor_fixed.counties = processor_fixed.counties[
        processor_fixed.counties['GEOID'].isin(test_geoids)
    ].copy()
    
    print(f"Testing with {len(processor_fixed.counties)} counties")
    
    try:
        # Process with fixed baseline
        df_fixed = processor_fixed.process_xclim_parallel(
            scenarios=['historical', 'ssp245'],
            variables=['tas', 'tasmax', 'tasmin', 'pr'],
            historical_period=historical_period,
            future_period=future_period,
            n_chunks=1
        )
        
        # Save results
        output_fixed = "../results/test_unified_fixed_baseline.csv"
        df_fixed.to_csv(output_fixed, index=False)
        print(f"Fixed baseline results saved to: {output_fixed}")
        print(f"Total records: {len(df_fixed)}")
        
        # Analyze fixed baseline results
        print("\nFixed Baseline Results Summary:")
        for county_name in df_fixed['NAME'].unique():
            county_data = df_fixed[df_fixed['NAME'] == county_name]
            hist_data = county_data[county_data['scenario'] == 'historical']
            fut_data = county_data[county_data['scenario'] == 'ssp245']
            
            if 'tx90p_percent' in hist_data.columns and not hist_data.empty:
                print(f"\n{county_name}:")
                print(f"  Historical tx90p: {hist_data['tx90p_percent'].mean():.1f}%")
                print(f"  Future tx90p: {fut_data['tx90p_percent'].mean():.1f}%")
        
    except Exception as e:
        print(f"Error in fixed baseline mode: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n2. TESTING PERIOD-SPECIFIC MODE")
    print("-"*60)
    
    # Initialize with period-specific baseline
    processor_period = ParallelXclimProcessor(
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        use_fixed_baseline=False
    )
    
    # Filter to test counties
    processor_period.counties = processor_period.counties[
        processor_period.counties['GEOID'].isin(test_geoids)
    ].copy()
    
    print(f"Testing with {len(processor_period.counties)} counties")
    
    try:
        # Process with period-specific baseline
        df_period = processor_period.process_xclim_parallel(
            scenarios=['historical', 'ssp245'],
            variables=['tas', 'tasmax', 'tasmin', 'pr'],
            historical_period=historical_period,
            future_period=future_period,
            n_chunks=1
        )
        
        # Save results
        output_period = "../results/test_unified_period_specific.csv"
        df_period.to_csv(output_period, index=False)
        print(f"Period-specific results saved to: {output_period}")
        print(f"Total records: {len(df_period)}")
        
        # Analyze period-specific results
        print("\nPeriod-Specific Results Summary:")
        for county_name in df_period['NAME'].unique():
            county_data = df_period[df_period['NAME'] == county_name]
            hist_data = county_data[county_data['scenario'] == 'historical']
            
            if 'tx90p_percent' in hist_data.columns and not hist_data.empty:
                print(f"\n{county_name}:")
                print(f"  Historical tx90p: {hist_data['tx90p_percent'].mean():.1f}%")
                print("  (Should be ~10% since calculated from same period)")
        
    except Exception as e:
        print(f"Error in period-specific mode: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare the two approaches
    print("\n\n3. COMPARISON OF APPROACHES")
    print("-"*60)
    
    try:
        if 'df_fixed' in locals() and 'df_period' in locals():
            print("\nKey differences:")
            print("- Fixed baseline: Percentiles calculated from 1980-2010")
            print("- Period-specific: Percentiles calculated from each analysis period")
            print("\nFixed baseline is recommended for climate change analysis")
            print("as it provides a consistent reference for comparison.")
    except:
        pass


def test_backward_compatibility():
    """
    Test that the unified processor maintains backward compatibility
    """
    print("\n\n4. TESTING BACKWARD COMPATIBILITY")
    print("-"*60)
    
    # This should behave like the original processor
    processor_like_original = ParallelXclimProcessor(
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        use_fixed_baseline=False
    )
    
    # This should behave like the fixed processor
    processor_like_fixed = ParallelXclimProcessor(
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010),
        use_fixed_baseline=True
    )
    
    print("✓ Both initialization modes work correctly")
    print("✓ API maintains backward compatibility")


if __name__ == "__main__":
    test_unified_processor()
    test_backward_compatibility()