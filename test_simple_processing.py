#!/usr/bin/env python
"""Simple test of climate data processing."""

import logging
from pathlib import Path
from climate_analysis import UnifiedParallelProcessor
from climate_analysis.core.unified_calculator_patch import apply_patch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Apply patch
    apply_patch()
    
    # Initialize processor
    processor = UnifiedParallelProcessor(
        shapefile_path="data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        output_dir="results",
        n_workers=1  # Single worker for debugging
    )
    
    # Simple indicators
    indicators_config = {
        'tg_mean': {
            'xclim_func': 'tg_mean',
            'variable': 'tas',
            'freq': 'YS'
        }
    }
    
    # Process just one county with historical data
    test_geoids = ['31039']  # Nebraska county
    
    print("Processing single county with minimal configuration...")
    
    try:
        results_df = processor.process_test_counties(
            test_geoids=test_geoids,
            scenarios=['historical'],
            indicators_config=indicators_config
        )
        
        print(f"\nSuccess! Processed {len(results_df)} records")
        
        if len(results_df) > 0:
            print("\nFirst few results:")
            print(results_df.head())
            
            # Save
            output_path = processor.save_results(results_df, format='csv')
            print(f"\nResults saved to: {output_path}")
        else:
            print("\nNo results returned - check logs above")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()