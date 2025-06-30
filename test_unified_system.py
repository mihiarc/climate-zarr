#!/usr/bin/env python3
"""
Simple end-to-end test of the unified climate processing system.

This script demonstrates:
1. Loading county data from shapefile
2. Processing climate indicators for selected counties
3. Saving results to CSV
"""

import sys
import time
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.unified_processor import UnifiedParallelProcessor
from src.utils.state_fips import get_state_name


def main():
    """Run a simple test of the unified system."""
    
    print("=" * 60)
    print("Unified Climate Processing System - End-to-End Test")
    print("=" * 60)
    
    # Configuration
    shapefile_path = "/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp"
    base_data_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    merged_baseline_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/merged_baselines/merged_baselines.pkl"
    output_dir = "test_results"
    
    # Test counties (small selection for quick test)
    test_counties = [
        '31039',  # Cuming County, NE
        '53069',  # Wahkiakum County, WA
        '48301',  # Loving County, TX (smallest county by population)
    ]
    
    # Climate scenarios to process
    scenarios = ['historical', 'ssp245']
    
    # Indicators to calculate
    indicators_config = {
        'tx90p': {
            'xclim_func': 'tx90p',
            'variable': 'tasmax',
            'freq': 'YS',
            'description': 'Days with max temp > 90th percentile'
        },
        'tn10p': {
            'xclim_func': 'tn10p',
            'variable': 'tasmin',
            'freq': 'YS',
            'description': 'Days with min temp < 10th percentile'
        },
        'tx_days_above_90F': {
            'xclim_func': 'tx_days_above',
            'variable': 'tasmax',
            'thresh': '305.37 K',
            'freq': 'YS',
            'description': 'Days with max temp > 90°F'
        },
        'precip_accumulation': {
            'xclim_func': 'precip_accumulation',
            'variable': 'pr',
            'freq': 'YS',
            'description': 'Annual precipitation'
        }
    }
    
    try:
        # Initialize the processor
        print("\n1. Initializing unified processor...")
        processor = UnifiedParallelProcessor(
            shapefile_path=shapefile_path,
            base_data_path=base_data_path,
            merged_baseline_path=merged_baseline_path,
            output_dir=output_dir,
            n_workers=2  # Use 2 workers for this test
        )
        print(f"   ✓ Loaded {len(processor.counties_gdf)} counties from shapefile")
        
        # Verify test counties exist
        test_counties_gdf = processor.counties_gdf[
            processor.counties_gdf['GEOID'].isin(test_counties)
        ]
        print(f"\n2. Test counties selected:")
        for _, county in test_counties_gdf.iterrows():
            state_name = get_state_name(county['STATEFP'])
            print(f"   - {county['NAME']} County, {state_name} (GEOID: {county['GEOID']})")
        
        # Process the counties
        print(f"\n3. Processing climate indicators...")
        print(f"   - Scenarios: {', '.join(scenarios)}")
        print(f"   - Indicators: {', '.join(indicators_config.keys())}")
        
        start_time = time.time()
        
        # Use the test_counties method for focused processing
        results_df = processor.process_test_counties(
            test_geoids=test_counties,
            scenarios=scenarios,
            indicators_config=indicators_config
        )
        
        elapsed = time.time() - start_time
        
        # Display results summary
        print(f"\n4. Processing complete in {elapsed:.1f} seconds")
        print(f"   ✓ Generated {len(results_df)} records")
        
        if not results_df.empty:
            # Show sample results
            print("\n5. Sample results:")
            print("-" * 60)
            
            # Group by county and scenario
            for geoid in test_counties:
                county_data = results_df[results_df['GEOID'] == geoid]
                if not county_data.empty:
                    county_name = county_data.iloc[0]['county_name']
                    print(f"\n{county_name} (GEOID: {geoid}):")
                    
                    for scenario in scenarios:
                        scenario_data = county_data[county_data['scenario'] == scenario]
                        if not scenario_data.empty:
                            # Show first and last year
                            years = scenario_data['year'].unique()
                            print(f"  {scenario}: {len(years)} years ({years.min()}-{years.max()})")
                            
                            # Show sample indicator values
                            sample_year = years[len(years)//2]  # Middle year
                            sample_row = scenario_data[scenario_data['year'] == sample_year].iloc[0]
                            print(f"    Sample values for {sample_year}:")
                            for ind in indicators_config.keys():
                                if ind in sample_row:
                                    value = sample_row[ind]
                                    if pd.notna(value):
                                        print(f"      - {ind}: {value:.2f}")
            
            # Save results
            print("\n6. Saving results...")
            output_path = processor.save_results(results_df, format='csv')
            print(f"   ✓ Results saved to: {output_path}")
            
            # Also save a summary
            summary_path = Path(output_dir) / "test_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("Unified Climate Processing System - Test Summary\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Counties processed: {len(test_counties)}\n")
                f.write(f"Scenarios: {', '.join(scenarios)}\n")
                f.write(f"Indicators: {', '.join(indicators_config.keys())}\n")
                f.write(f"Total records: {len(results_df)}\n")
                f.write(f"Processing time: {elapsed:.1f} seconds\n")
                f.write(f"Output file: {output_path}\n")
            
            print(f"   ✓ Summary saved to: {summary_path}")
            
        else:
            print("\n⚠ Warning: No results generated!")
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())