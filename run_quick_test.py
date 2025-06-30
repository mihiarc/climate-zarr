#!/usr/bin/env python
"""Quick test with just 2 years of data."""

import logging
from pathlib import Path
from climate_analysis import UnifiedParallelProcessor
from climate_analysis.core.unified_calculator_patch import apply_patch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    # Apply patch
    apply_patch()
    
    # Initialize processor
    processor = UnifiedParallelProcessor(
        shapefile_path="data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        output_dir="results",
        n_workers=1
    )
    
    # Just one simple indicator
    indicators_config = {
        'tg_mean': {
            'xclim_func': 'tg_mean',
            'variable': 'tas',
            'freq': 'YS'
        }
    }
    
    # One county, one scenario
    print("Processing 1 county with 1 indicator...")
    
    results_df = processor.process_test_counties(
        test_geoids=['31039'],
        scenarios=['historical'],
        indicators_config=indicators_config
    )
    
    if len(results_df) > 0:
        print(f"\nSuccess! Got {len(results_df)} results")
        print("\nSample results:")
        print(results_df[['GEOID', 'NAME', 'year', 'tg_mean_C']].head(10))
        
        # Save
        output_file = processor.save_results(results_df, format='csv')
        print(f"\nSaved to: {output_file}")
    else:
        print("\nNo results returned")

if __name__ == "__main__":
    main()