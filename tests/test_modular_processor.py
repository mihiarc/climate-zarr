#!/usr/bin/env python3
"""
Test the modular climate processor design
"""

import pandas as pd
import sys
import time
sys.path.append('../src')
from parallel_processor import ParallelClimateProcessor
from climate_indicator_calculator import ClimateIndicatorCalculator


def test_core_calculator():
    """Test the core climate indicator calculator independently."""
    print("TESTING CORE CLIMATE CALCULATOR")
    print("="*60)
    
    # Initialize calculator
    calculator = ClimateIndicatorCalculator(
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010)
    )
    
    # Test county info for Cuming, NE
    county_info = {
        'geoid': '31039',
        'name': 'Cuming',
        'state': '31',
        'bounds': (-96.7887, 41.7193, -96.1251, 42.2088)  # Example bounds
    }
    
    print(f"\nTesting with {county_info['name']} county...")
    
    try:
        # Test baseline calculation
        print("\n1. Testing baseline percentile calculation...")
        thresholds = calculator.calculate_baseline_percentiles(county_info['bounds'])
        
        if 'tasmax_p90_doy' in thresholds:
            print(f"   ✓ Calculated tasmax 90th percentile baseline")
            print(f"     Shape: {thresholds['tasmax_p90_doy'].shape}")
        
        if 'tasmin_p10_doy' in thresholds:
            print(f"   ✓ Calculated tasmin 10th percentile baseline")
            print(f"     Shape: {thresholds['tasmin_p10_doy'].shape}")
            
        # Test single county processing
        print("\n2. Testing full county processing...")
        results = calculator.process_county(
            county_info=county_info,
            scenarios=['historical'],
            variables=['tas', 'tasmax', 'tasmin', 'pr'],
            historical_period=(2009, 2010),  # Just 2 years for quick test
            future_period=(2040, 2041)
        )
        
        print(f"   ✓ Generated {len(results)} records")
        
        if results:
            # Check first record
            first = results[0]
            print(f"\n   Sample record:")
            print(f"     County: {first['NAME']}, {first['STATE']}")
            print(f"     Year: {first['year']}")
            print(f"     Mean temp: {first.get('tg_mean_C', 'N/A'):.1f}°C")
            print(f"     Days >90°F: {first.get('tx_days_above_90F', 'N/A')}")
            
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_parallel_processor():
    """Test the parallel processor."""
    print("\n\nTESTING PARALLEL PROCESSOR")
    print("="*60)
    
    # Initialize processor
    processor = ParallelClimateProcessor(
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010)
    )
    
    # Test counties
    test_geoids = ['31039', '53069', '48453']  # Cuming NE, Wahkiakum WA, Travis TX
    
    print(f"\nTesting with {len(test_geoids)} counties...")
    
    # Define progress callback
    def progress_update(completed, total, elapsed):
        print(f"  Progress callback: {completed}/{total} batches, {elapsed:.1f}s elapsed")
    
    try:
        # Test with multiple workers
        df = processor.process_test_counties(
            test_geoids=test_geoids,
            scenarios=['historical', 'ssp245'],
            variables=['tas', 'tasmax', 'tasmin', 'pr'],
            historical_period=(2009, 2010),
            future_period=(2040, 2041),
            n_workers=2,  # Use 2 workers
            progress_callback=progress_update
        )
        
        # Save results
        output_file = "../results/test_modular_processor.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Results saved to: {output_file}")
        print(f"  Total records: {len(df)}")
        
        # Analyze results
        print("\nResults summary by county:")
        for geoid in test_geoids:
            county_data = df[df['GEOID'] == geoid]
            if not county_data.empty:
                name = county_data.iloc[0]['NAME']
                n_records = len(county_data)
                scenarios = county_data['scenario'].unique()
                print(f"  {name}: {n_records} records, scenarios: {', '.join(scenarios)}")
                
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_backward_compatibility():
    """Test backward compatibility with the wrapper."""
    print("\n\nTESTING BACKWARD COMPATIBILITY")
    print("="*60)
    
    from parallel_xclim_processor import ParallelXclimProcessor
    
    # Initialize using old API
    processor = ParallelXclimProcessor(
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010)
    )
    
    # Filter counties (old API style)
    processor.counties = processor.counties[
        processor.counties['GEOID'].isin(['31039'])
    ].copy()
    
    print(f"\nTesting backward compatibility with {len(processor.counties)} county...")
    
    try:
        # Use old API
        df = processor.process_xclim_parallel(
            scenarios=['historical'],
            historical_period=(2009, 2010),
            future_period=(2040, 2041),
            n_chunks=1
        )
        
        print(f"✓ Backward compatibility works: {len(df)} records generated")
        
    except Exception as e:
        print(f"✗ Backward compatibility error: {e}")


def test_performance_comparison():
    """Compare performance of modular vs monolithic design."""
    print("\n\nPERFORMANCE COMPARISON")
    print("="*60)
    
    test_geoids = ['31039', '53069']  # 2 counties for quick comparison
    
    # Test modular processor
    print("\nModular processor:")
    processor = ParallelClimateProcessor(
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010)
    )
    
    start = time.time()
    df_modular = processor.process_test_counties(
        test_geoids=test_geoids,
        scenarios=['historical'],
        historical_period=(2009, 2010),
        future_period=(2040, 2041),
        n_workers=1
    )
    modular_time = time.time() - start
    
    print(f"  Time: {modular_time:.2f}s")
    print(f"  Records: {len(df_modular)}")
    print(f"  Time per county: {modular_time/len(test_geoids):.2f}s")


if __name__ == "__main__":
    # Run all tests
    test_core_calculator()
    test_parallel_processor()
    test_backward_compatibility()
    test_performance_comparison()
    
    print("\n" + "="*60)
    print("All tests complete!")