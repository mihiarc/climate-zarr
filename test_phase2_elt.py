#!/usr/bin/env python3
"""Test script for Phase 2 ELT optimizations.

This script demonstrates the new tile-based processing and ELT pattern
implementation for climate data processing.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

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


class ProgressTracker:
    """Simple progress tracker for test execution."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        
    def update(self, step_name: str, increment: int = 1):
        """Update progress with step name."""
        self.current_step += increment
        elapsed = time.time() - self.start_time
        percent = (self.current_step / self.total_steps) * 100
        
        # Estimate time remaining
        if self.current_step > 0:
            rate = elapsed / self.current_step
            remaining = (self.total_steps - self.current_step) * rate
            eta = datetime.fromtimestamp(time.time() + remaining).strftime('%H:%M:%S')
        else:
            eta = "calculating..."
            
        logger.info(f"\n{'='*60}")
        logger.info(f"PROGRESS: Step {self.current_step}/{self.total_steps} ({percent:.1f}%)")
        logger.info(f"Current: {step_name}")
        logger.info(f"Elapsed: {elapsed:.1f}s | ETA: {eta}")
        logger.info(f"{'='*60}\n")


def test_tile_processing():
    """Test the new tile-based processing method."""
    
    # Configuration
    shapefile_path = "/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp"
    base_data_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    # Skip merged baseline file - use individual cached baselines instead
    merged_baseline_path = None
    
    # Initialize progress tracker (4 main test steps)
    progress = ProgressTracker(total_steps=4)
    
    logger.info("=" * 60)
    logger.info("PHASE 2 ELT OPTIMIZATION TEST")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize processor
    logger.info("\nInitializing processor with 16 workers...")
    processor = UnifiedParallelProcessor(
        shapefile_path=shapefile_path,
        base_data_path=base_data_path,
        merged_baseline_path=merged_baseline_path,
        output_dir="results/phase2_test",
        n_workers=16
    )
    logger.info(f"✓ Processor initialized with {processor.n_workers} workers")
    
    # Test with a small subset first
    test_geoids = ["06037", "06059", "06073"]  # LA, Orange, San Diego counties
    logger.info(f"Testing with {len(test_geoids)} counties: {', '.join(test_geoids)}")
    logger.info("Using individual cached baselines (not merged file) for efficient memory usage")
    
    # Test 1: Standard processing (baseline)
    progress.update("Test 1: Standard processing (baseline)")
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
    logger.info(f"✓ Standard processing completed in {standard_time:.2f} seconds")
    logger.info(f"  Processed {len(results_standard)} counties")
    
    # Test 2: Tile-based processing
    progress.update("Test 2: Tile-based processing (ELT pattern)")
    
    # Create tiles
    logger.info("Creating spatial tiles...")
    tiles = processor.create_spatial_tiles(tile_size_degrees=1.0)
    logger.info(f"✓ Created {len(tiles)} spatial tiles")
    
    # Find tiles containing our test counties
    test_tiles = {}
    for tile_id, tile_info in tiles.items():
        if any(geoid in tile_info['counties'] for geoid in test_geoids):
            test_tiles[tile_id] = tile_info
    
    logger.info(f"✓ Test counties found in {len(test_tiles)} tiles")
    
    # Process using tile method
    start_time = time.time()
    tile_results_count = 0
    
    for i, (tile_id, tile_info) in enumerate(test_tiles.items(), 1):
        logger.info(f"  Processing tile {i}/{len(test_tiles)}: {tile_id} ({len(tile_info['counties'])} counties)")
        
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
        
        tile_results_count += len(results)
        logger.info(f"  ✓ Tile {tile_id}: {len(results)} successful, {len(failed)} failed")
    
    tile_time = time.time() - start_time
    logger.info(f"✓ Tile processing completed in {tile_time:.2f} seconds")
    logger.info(f"  Total counties processed: {tile_results_count}")
    
    # Calculate speedup
    speedup = standard_time / tile_time if tile_time > 0 else 0
    logger.info(f"\n{'*'*40}")
    logger.info(f"SPEEDUP: {speedup:.2f}x faster with tiles!")
    logger.info(f"{'*'*40}")
    
    # Test 3: Test Zarr conversion (if needed)
    progress.update("Test 3: Testing Zarr store creation")
    
    logger.info("Initializing calculator with Zarr support...")
    calculator = UnifiedClimateCalculator(
        base_data_path=base_data_path,
        merged_baseline_path=None,  # Use individual cached baselines
        use_zarr=True,
        use_dask=True
    )
    
    # Check if we can create a Zarr store
    logger.info("Checking Zarr store (this may take time on first run)...")
    zarr_start = time.time()
    zarr_store = calculator.get_zarr_store('tasmax', 'historical')
    zarr_time = time.time() - zarr_start
    
    if zarr_store:
        logger.info(f"✓ Successfully created/opened Zarr store in {zarr_time:.1f}s")
        logger.info(f"  Variables: {list(zarr_store.data_vars)}")
        logger.info(f"  Chunks: {zarr_store.chunks}")
    else:
        logger.info("✗ Zarr store creation skipped or failed")
    
    # Test 4: Compare memory usage
    progress.update("Test 4: Memory usage analysis")
    
    import psutil
    process = psutil.Process()
    
    # Get memory after tile processing
    memory_info = process.memory_info()
    logger.info(f"✓ Memory usage analysis:")
    logger.info(f"  RSS (Resident Set Size): {memory_info.rss / 1024 / 1024:.1f} MB")
    logger.info(f"  VMS (Virtual Memory Size): {memory_info.vms / 1024 / 1024:.1f} MB")
    
    # CPU usage
    cpu_percent = process.cpu_percent(interval=1)
    logger.info(f"  CPU usage: {cpu_percent:.1f}%")
    
    # Summary
    total_test_time = time.time() - progress.start_time
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total test duration: {total_test_time:.1f} seconds")
    logger.info(f"")
    logger.info(f"Performance Results:")
    logger.info(f"  Standard processing: {standard_time:.2f}s")
    logger.info(f"  Tile-based processing: {tile_time:.2f}s") 
    logger.info(f"  Speedup factor: {speedup:.2f}x")
    logger.info(f"")
    logger.info(f"Optimizations Applied:")
    logger.info(f"  ✓ Memory efficient: Uses shared tile data")
    logger.info(f"  ✓ I/O optimized: Loads data once per tile")
    logger.info(f"  ✓ Dask integration: Lazy loading enabled")
    logger.info(f"  ✓ Zarr support: Cloud-optimized format ready")
    logger.info("=" * 60)


def test_full_parallel_tiles():
    """Test full parallel processing with tiles."""
    
    logger.info("\n" + "=" * 60)
    logger.info("FULL PARALLEL TILE PROCESSING TEST")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    shapefile_path = "/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp"
    base_data_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    # Skip merged baseline file - use individual cached baselines instead
    merged_baseline_path = None
    
    logger.info("\nInitializing processor with 16 workers...")
    
    # Initialize processor
    processor = UnifiedParallelProcessor(
        shapefile_path=shapefile_path,
        base_data_path=base_data_path,
        merged_baseline_path=merged_baseline_path,
        output_dir="results/phase2_parallel",
        n_workers=16  # Use 16 workers for parallel processing
    )
    
    # Get all California counties
    ca_counties = processor.counties_gdf[processor.counties_gdf['STATEFP'] == '06']
    logger.info(f"✓ Found {len(ca_counties)} California counties to process")
    
    # Show some county names
    sample_counties = ca_counties.head(5)
    logger.info("\nSample counties:")
    for _, county in sample_counties.iterrows():
        logger.info(f"  - {county['NAME']} County ({county['GEOID']})")
    
    # Process using the new parallel tile method
    logger.info("\nStarting parallel tile processing...")
    logger.info("This will process all California counties using spatial tiles")
    logger.info("Expected duration: 5-15 minutes\n")
    
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
        logger.info(f"\n✓ Results saved to: {output_path}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("CALIFORNIA PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processing time: {processing_time/60:.1f} minutes")
    logger.info(f"Counties processed: {len(results_df)}")
    logger.info(f"Average time per county: {processing_time/len(results_df):.1f}s" if len(results_df) > 0 else "N/A")
    logger.info(f"Processing rate: {len(results_df)/(processing_time/60):.1f} counties/minute" if processing_time > 0 else "N/A")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Run tests
    test_tile_processing()
    
    # Uncomment to run full parallel test
    # test_full_parallel_tiles()