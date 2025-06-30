#!/usr/bin/env python
"""Run climate analysis with proper data structure."""

import logging
from pathlib import Path
from climate_analysis import UnifiedParallelProcessor
from climate_analysis.core.unified_calculator_patch import apply_patch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Apply patch to handle variable/scenario directory structure
    apply_patch()
    
    # Check if we need to use merged baselines
    merged_baseline_path = None
    merged_baselines_dir = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/merged_baselines")
    if merged_baselines_dir.exists():
        baseline_files = list(merged_baselines_dir.glob("*.pkl"))
        if baseline_files:
            merged_baseline_path = str(baseline_files[0])
            print(f"Using merged baseline: {merged_baseline_path}")
    
    # Initialize processor
    processor = UnifiedParallelProcessor(
        shapefile_path="data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        merged_baseline_path=merged_baseline_path,
        output_dir="results",
        n_workers=4
    )
    
    # Define indicators to calculate
    indicators_config = {
        'tx90p': {
            'xclim_func': 'tx90p',
            'variable': 'tasmax',
            'freq': 'YS'
        },
        'tn10p': {
            'xclim_func': 'tn10p', 
            'variable': 'tasmin',
            'freq': 'YS'
        },
        'tg_mean': {
            'xclim_func': 'tg_mean',
            'variable': 'tas',
            'freq': 'YS'
        },
        'tx_days_above_90F': {
            'xclim_func': 'tx_days_above',
            'variable': 'tasmax',
            'thresh': '305.37 K',
            'freq': 'YS'
        },
        'tn_days_below_32F': {
            'xclim_func': 'tn_days_below',
            'variable': 'tasmin',
            'thresh': '273.15 K',
            'freq': 'YS'
        },
        'precip_accumulation': {
            'xclim_func': 'precip_accumulation',
            'variable': 'pr',
            'freq': 'YS'
        },
        'days_precip_over_25.4mm': {
            'xclim_func': 'wetdays',
            'variable': 'pr',
            'thresh': '0.000294 kg m-2 s-1',
            'freq': 'YS'
        }
    }
    
    # Process a few test counties
    # Nebraska (31039) and Washington (53069) counties
    test_geoids = ['31039', '53069']
    
    print(f"\nProcessing {len(test_geoids)} test counties...")
    print(f"Counties: {test_geoids}")
    print(f"Scenarios: historical, ssp245")
    print(f"Indicators: {list(indicators_config.keys())}")
    
    try:
        results_df = processor.process_test_counties(
            test_geoids=test_geoids,
            scenarios=['historical', 'ssp245'],
            indicators_config=indicators_config
        )
        
        # Save results
        output_path = processor.save_results(results_df, format='parquet')
        print(f"\n✓ Results saved to: {output_path}")
        
        # Also save as CSV for easy viewing
        csv_path = output_path.with_suffix('.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"✓ CSV version saved to: {csv_path}")
        
        # Show summary
        print(f"\nResults summary:")
        print(f"- Total records: {len(results_df)}")
        print(f"- Counties processed: {results_df['GEOID'].nunique()}")
        print(f"- Scenarios: {results_df['scenario'].unique().tolist()}")
        print(f"- Years covered: {results_df['year'].min()} - {results_df['year'].max()}")
        
        # Show sample results
        print("\nSample results (first 5 rows):")
        print(results_df.head())
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()