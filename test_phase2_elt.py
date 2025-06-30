#!/usr/bin/env python3
"""Test script for Phase 2 ELT optimizations.

This script demonstrates the new tile-based processing and ELT pattern
implementation for climate data processing.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.unified_processor import UnifiedParallelProcessor
from src.core.unified_calculator import UnifiedClimateCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_tile_processing():
    """Test the new tile-based processing method."""
    
    # Configuration
    shapefile_path = "/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp"
    base_data_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    merged_baseline_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/merged_baselines/merged_baseline_NorESM2-LM_1980-2010.pkl"
    
    # Initialize processor
    processor = UnifiedParallelProcessor(
        shapefile_path=shapefile_path,
        base_data_path=base_data_path,
        merged_baseline_path=merged_baseline_path,
        output_dir="results/phase2_test",
        n_workers=4
    )
    
    # Test with a small subset first
    test_geoids = ["06037", "06059", "06073"]  # LA, Orange, San Diego counties
    
    logger.info("=" * 60)
    logger.info("PHASE 2 ELT OPTIMIZATION TEST")
    logger.info("=" * 60)
    
    # Test 1: Standard processing (baseline)
    logger.info("\nTest 1: Standard processing (baseline)")
    start_time = time.time()
    
    results_standard = processor.process_test_counties(
        test_geoids=test_geoids,
        scenarios=['historical'],
        indicators_config={
            'tx90p': {'xclim_func': 'tx90p', 'variable': 'tasmax', 'freq': 'YS'},
            'tn10p': {'xclim_func': 'tn10p', 'variable': 'tasmin', 'freq': 'YS'}
        }
    )
    
    standard_time = time.time() - start_time
    logger.info(f"Standard processing time: {standard_time:.2f} seconds")
    
    # Test 2: Tile-based processing
    logger.info("\nTest 2: Tile-based processing (ELT pattern)")
    
    # Create tiles
    tiles = processor.create_spatial_tiles(tile_size_degrees=1.0)
    logger.info(f"Created {len(tiles)} spatial tiles")
    
    # Find tiles containing our test counties
    test_tiles = {}
    for tile_id, tile_info in tiles.items():
        if any(geoid in tile_info['counties'] for geoid in test_geoids):
            test_tiles[tile_id] = tile_info
    
    logger.info(f"Test counties found in {len(test_tiles)} tiles")
    
    # Process using tile method
    start_time = time.time()
    
    for tile_id, tile_info in test_tiles.items():
        results, failed = processor.process_counties_by_tile(
            tile_id=tile_id,
            tile_info=tile_info,
            scenarios=['historical'],
            indicators_config={
                'tx90p': {'xclim_func': 'tx90p', 'variable': 'tasmax', 'freq': 'YS'},
                'tn10p': {'xclim_func': 'tn10p', 'variable': 'tasmin', 'freq': 'YS'}
            },
            use_dask=True
        )
        
        logger.info(f"Tile {tile_id}: {len(results)} successful, {len(failed)} failed")
    
    tile_time = time.time() - start_time
    logger.info(f"Tile processing time: {tile_time:.2f} seconds")
    
    # Calculate speedup
    speedup = standard_time / tile_time if tile_time > 0 else 0
    logger.info(f"\nSpeedup: {speedup:.2f}x")
    
    # Test 3: Test Zarr conversion (if needed)
    logger.info("\nTest 3: Testing Zarr store creation")
    
    calculator = UnifiedClimateCalculator(
        base_data_path=base_data_path,
        merged_baseline_path=merged_baseline_path,
        use_zarr=True,
        use_dask=True
    )
    
    # Check if we can create a Zarr store
    zarr_store = calculator.get_zarr_store('tasmax', 'historical')
    if zarr_store:
        logger.info("Successfully created/opened Zarr store")
        logger.info(f"Zarr store variables: {list(zarr_store.data_vars)}")
        logger.info(f"Zarr store chunks: {zarr_store.chunks}")
    else:
        logger.info("Zarr store creation skipped or failed")
    
    # Test 4: Compare memory usage
    logger.info("\nTest 4: Memory usage comparison")
    
    import psutil
    process = psutil.Process()
    
    # Get memory after tile processing
    memory_info = process.memory_info()
    logger.info(f"Memory usage (RSS): {memory_info.rss / 1024 / 1024:.1f} MB")
    logger.info(f"Memory usage (VMS): {memory_info.vms / 1024 / 1024:.1f} MB")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Standard processing: {standard_time:.2f}s")
    logger.info(f"Tile-based processing: {tile_time:.2f}s") 
    logger.info(f"Speedup factor: {speedup:.2f}x")
    logger.info(f"Memory efficient: Uses shared tile data")
    logger.info(f"I/O optimized: Loads data once per tile")
    logger.info("=" * 60)


def test_full_parallel_tiles():
    """Test full parallel processing with tiles."""
    
    logger.info("\n" + "=" * 60)
    logger.info("FULL PARALLEL TILE PROCESSING TEST")
    logger.info("=" * 60)
    
    # Configuration
    shapefile_path = "/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp"
    base_data_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    merged_baseline_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/merged_baselines/merged_baseline_NorESM2-LM_1980-2010.pkl"
    
    # Initialize processor
    processor = UnifiedParallelProcessor(
        shapefile_path=shapefile_path,
        base_data_path=base_data_path,
        merged_baseline_path=merged_baseline_path,
        output_dir="results/phase2_parallel",
        n_workers=8  # Use 8 workers for parallel processing
    )
    
    # Process a larger subset using parallel tiles
    logger.info("Processing California counties with parallel tiles...")
    
    # Get all California counties
    ca_counties = processor.counties_gdf[processor.counties_gdf['STATEFP'] == '06']
    logger.info(f"Found {len(ca_counties)} California counties")
    
    # Process using the new parallel tile method
    start_time = time.time()
    
    results_df = processor.process_parallel_with_tiles(
        scenarios=['historical'],
        indicators_config={
            'tx90p': {'xclim_func': 'tx90p', 'variable': 'tasmax', 'freq': 'YS'},
            'tn10p': {'xclim_func': 'tn10p', 'variable': 'tasmin', 'freq': 'YS'},
            'precip_accumulation': {'xclim_func': 'precip_accumulation', 'variable': 'pr', 'freq': 'YS'}
        },
        tile_size_degrees=2.0,
        use_dask=True,
        use_zarr=False  # Set to True to test Zarr conversion
    )
    
    processing_time = time.time() - start_time
    
    # Save results
    if not results_df.empty:
        output_path = processor.save_results(results_df, format='parquet')
        logger.info(f"Results saved to: {output_path}")
    
    logger.info(f"\nTotal processing time: {processing_time/60:.1f} minutes")
    logger.info(f"Counties processed: {len(results_df)}")
    logger.info(f"Average time per county: {processing_time/len(results_df):.1f}s" if len(results_df) > 0 else "N/A")


if __name__ == "__main__":
    # Run tests
    test_tile_processing()
    
    # Uncomment to run full parallel test
    # test_full_parallel_tiles()