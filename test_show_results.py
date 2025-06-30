#!/usr/bin/env python3
"""Simple test to show climate indicator results for 3 counties."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.unified_processor import UnifiedParallelProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    # Configuration
    shapefile_path = "/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp"
    base_data_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    
    # Initialize processor
    logger.info("Initializing processor...")
    processor = UnifiedParallelProcessor(
        shapefile_path=shapefile_path,
        base_data_path=base_data_path,
        output_dir="results/phase2_test",
        n_workers=4  # Use fewer workers for quick test
    )
    
    # Test with 3 counties
    test_geoids = ["06037", "06059", "06073"]  # LA, Orange, San Diego counties
    logger.info(f"Processing {len(test_geoids)} counties: {', '.join(test_geoids)}")
    
    # Process counties
    results = processor.process_test_counties(
        test_geoids=test_geoids,
        scenarios=['historical'],
        indicators_config={
            'tx90p': {'xclim_func': 'tx90p', 'variable': 'tasmax', 'freq': 'YS'},
            'tn10p': {'xclim_func': 'tn10p', 'variable': 'tasmin', 'freq': 'YS'},
            'tg_mean': {'xclim_func': 'tg_mean', 'variable': 'tas', 'freq': 'YS'}
        },
        year_range=(2000, 2010)
    )
    
    # Display results
    if not results.empty:
        logger.info(f"\n✓ Successfully processed {len(results)} records")
        logger.info("\nSample results (first 5 rows):")
        print(results.head())
        
        # Show summary statistics
        logger.info("\nSummary by county:")
        for geoid in test_geoids:
            county_data = results[results['GEOID'] == geoid]
            if not county_data.empty:
                county_name = county_data.iloc[0]['NAME']
                logger.info(f"\n{county_name} County ({geoid}):")
                logger.info(f"  Years: {county_data['Year'].min()} - {county_data['Year'].max()}")
                logger.info(f"  Mean temperature (tg_mean): {county_data['tg_mean'].mean():.2f}°C")
                if 'tx90p' in county_data.columns:
                    logger.info(f"  Hot days (tx90p): {county_data['tx90p'].mean():.1f} days/year")
                if 'tn10p' in county_data.columns:
                    logger.info(f"  Cold nights (tn10p): {county_data['tn10p'].mean():.1f} days/year")
        
        # Save results
        output_file = Path("results/phase2_test/county_results.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
        logger.info(f"\n✓ Results saved to: {output_file}")
    else:
        logger.error("No results returned!")

if __name__ == "__main__":
    main()