"""Unified parallel processor for climate data.

This module provides the production parallel processing implementation,
incorporating the best practices from our optimization efforts.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pandas as pd
import geopandas as gpd

from .unified_calculator import UnifiedClimateCalculator
from ..utils.file_operations import ensure_directory, save_json
from ..utils.state_fips import get_state_name

logger = logging.getLogger(__name__)


class UnifiedParallelProcessor:
    """Production parallel processor for climate indicators.
    
    Features:
    - Dynamic batch sizing based on available resources
    - Progress tracking with ETA
    - Error recovery at batch level
    - Optimized for merged baseline strategy
    """
    
    def __init__(
        self,
        shapefile_path: str,
        base_data_path: str,
        merged_baseline_path: Optional[str] = None,
        output_dir: str = "results",
        n_workers: Optional[int] = None
    ):
        """Initialize the parallel processor.
        
        Args:
            shapefile_path: Path to counties shapefile
            base_data_path: Path to climate data
            merged_baseline_path: Path to merged baseline file
            output_dir: Directory for output files
            n_workers: Number of worker processes (None for auto)
        """
        self.shapefile_path = Path(shapefile_path)
        self.base_data_path = Path(base_data_path)
        self.merged_baseline_path = merged_baseline_path
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers or mp.cpu_count()
        
        # Load counties
        self.counties_gdf = self._load_counties()
        
        # Ensure output directory exists
        ensure_directory(self.output_dir)
    
    def _load_counties(self) -> gpd.GeoDataFrame:
        """Load and prepare counties shapefile."""
        gdf = gpd.read_file(self.shapefile_path)
        
        # Add bounds for each county
        gdf['bounds'] = gdf.geometry.bounds.values.tolist()
        
        return gdf
    
    def prepare_county_info(self, county_row) -> Dict[str, Any]:
        """Prepare county information for processing.
        
        Args:
            county_row: Row from counties GeoDataFrame
            
        Returns:
            Dictionary with county information including bounds
        """
        bounds = county_row.geometry.bounds
        
        return {
            'GEOID': county_row['GEOID'],
            'NAME': county_row['NAME'],
            'STATE': get_state_name(county_row.get('STATEFP', '')),
            'bounds': list(bounds)  # (minx, miny, maxx, maxy)
        }
    
    def process_test_counties(
        self,
        test_geoids: List[str],
        scenarios: List[str],
        indicators_config: Dict[str, Dict[str, Any]],
        **kwargs
    ) -> pd.DataFrame:
        """Process a small set of test counties.
        
        Args:
            test_geoids: List of GEOIDs to process
            scenarios: List of climate scenarios
            indicators_config: Configuration for indicators
            **kwargs: Additional arguments passed to process_all_counties
            
        Returns:
            DataFrame with results for test counties
        """
        # Validate test counties exist
        test_counties = self.counties_gdf[self.counties_gdf['GEOID'].isin(test_geoids)]
        
        if len(test_counties) == 0:
            raise ValueError(f"No counties found with GEOIDs: {test_geoids}")
        
        if len(test_counties) < len(test_geoids):
            found_geoids = test_counties['GEOID'].tolist()
            missing = [g for g in test_geoids if g not in found_geoids]
            logger.warning(f"Some GEOIDs not found: {missing}")
        
        logger.info(f"Processing {len(test_counties)} test counties")
        
        # Process the test counties
        return self.process_all_counties(
            scenarios=scenarios,
            indicators_config=indicators_config,
            counties_filter=test_geoids,
            **kwargs
        )
    
    def process_all_counties(
        self,
        scenarios: List[str],
        indicators_config: Dict[str, Dict[str, Any]],
        counties_filter: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Process all counties in parallel.
        
        Args:
            scenarios: List of climate scenarios
            indicators_config: Configuration for indicators
            counties_filter: Optional list of county GEOIDs to process
            batch_size: Counties per batch (None for auto)
            progress_callback: Optional callback function(completed, total, elapsed)
            variables: Optional list of variables (for backward compatibility)
            
        Returns:
            DataFrame with all results
        """
        # Handle backward compatibility - convert variables list to indicators_config
        if variables and not indicators_config:
            logger.warning("Using deprecated 'variables' parameter. Please use 'indicators_config' instead.")
            indicators_config = self._create_default_indicators_config(variables)
        # Filter counties if requested
        if counties_filter:
            counties_to_process = self.counties_gdf[
                self.counties_gdf['GEOID'].isin(counties_filter)
            ]
        else:
            counties_to_process = self.counties_gdf
        
        total_counties = len(counties_to_process)
        logger.info(f"Processing {total_counties} counties with {self.n_workers} workers")
        
        # Create batches
        if batch_size is None:
            # Auto batch size: aim for ~10 batches per worker
            batch_size = max(1, total_counties // (self.n_workers * 10))
        
        batches = self._create_batches(counties_to_process, batch_size)
        logger.info(f"Created {len(batches)} batches of size ~{batch_size}")
        
        # Process batches in parallel
        all_results = []
        failed_counties = []
        
        start_time = time.time()
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(
                    self._process_batch,
                    batch,
                    scenarios,
                    indicators_config,
                    self.base_data_path,
                    self.merged_baseline_path
                ): i
                for i, batch in enumerate(batches)
            }
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_results, batch_failed = future.result()
                    all_results.extend(batch_results)
                    failed_counties.extend(batch_failed)
                    
                    completed += len(batches[batch_idx])
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total_counties - completed) / rate if rate > 0 else 0
                    
                    logger.info(
                        f"Progress: {completed}/{total_counties} counties "
                        f"({completed/total_counties*100:.1f}%) - "
                        f"ETA: {eta/60:.1f} minutes"
                    )
                    
                    if progress_callback:
                        progress_callback(completed, total_counties, elapsed)
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    failed_counties.extend([c['GEOID'] for c in batches[batch_idx]])
        
        # Report summary
        total_time = time.time() - start_time
        logger.info(
            f"Completed processing in {total_time/60:.1f} minutes. "
            f"Success: {len(all_results)} records, "
            f"Failed: {len(failed_counties)} counties"
        )
        
        if failed_counties:
            logger.warning(f"Failed counties: {failed_counties}")
            save_json(
                {'failed_counties': failed_counties},
                self.output_dir / 'failed_counties.json'
            )
        
        return pd.DataFrame(all_results)
    
    @staticmethod
    def _process_batch(
        counties: List[Dict[str, Any]],
        scenarios: List[str],
        indicators_config: Dict[str, Dict[str, Any]],
        base_data_path: str,
        merged_baseline_path: Optional[str]
    ) -> Tuple[List[Dict], List[str]]:
        """Process a batch of counties (worker function).
        
        Returns:
            Tuple of (results_list, failed_county_ids)
        """
        # Create calculator for this worker
        calculator = UnifiedClimateCalculator(
            base_data_path=base_data_path,
            merged_baseline_path=merged_baseline_path
        )
        
        batch_results = []
        failed_counties = []
        
        for county in counties:
            try:
                county_info = {
                    'geoid': county['GEOID'],
                    'name': county['NAME'],
                    'state': county.get('STATE', 'Unknown')  # Already processed by prepare_county_info
                }
                
                results = calculator.calculate_indicators(
                    scenarios=scenarios,
                    county_bounds=county['bounds'],
                    county_info=county_info,
                    indicators_config=indicators_config
                )
                
                batch_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error processing county {county['GEOID']}: {e}")
                failed_counties.append(county['GEOID'])
        
        return batch_results, failed_counties
    
    def _create_batches(
        self,
        counties_gdf: gpd.GeoDataFrame,
        batch_size: int
    ) -> List[List[Dict]]:
        """Create batches of counties for processing."""
        counties_list = counties_gdf.to_dict('records')
        
        batches = []
        for i in range(0, len(counties_list), batch_size):
            batches.append(counties_list[i:i + batch_size])
        
        return batches
    
    def _create_default_indicators_config(self, variables: List[str]) -> Dict[str, Dict[str, Any]]:
        """Create default indicators configuration from variables list.
        
        For backward compatibility with old API.
        """
        default_indicators = {
            'tx90p': {'xclim_func': 'tx90p', 'variable': 'tasmax', 'freq': 'YS'},
            'tn10p': {'xclim_func': 'tn10p', 'variable': 'tasmin', 'freq': 'YS'},
            'tx_days_above_90F': {'xclim_func': 'tx_days_above', 'variable': 'tasmax', 
                                  'thresh': '305.37 K', 'freq': 'YS'},
            'tn_days_below_32F': {'xclim_func': 'tn_days_below', 'variable': 'tasmin',
                                  'thresh': '273.15 K', 'freq': 'YS'},
            'tg_mean': {'xclim_func': 'tg_mean', 'variable': 'tas', 'freq': 'YS'},
            'precip_accumulation': {'xclim_func': 'precip_accumulation', 'variable': 'pr', 'freq': 'YS'},
            'days_precip_over_25.4mm': {'xclim_func': 'wetdays', 'variable': 'pr',
                                        'thresh': '0.000294 kg m-2 s-1', 'freq': 'YS'}
        }
        
        # Filter to only include indicators for requested variables
        config = {}
        for ind_name, ind_config in default_indicators.items():
            if ind_config.get('variable') in variables:
                config[ind_name] = ind_config
        
        return config
    
    def process_parallel(
        self,
        scenarios: List[str] = ['historical', 'ssp245'],
        variables: List[str] = ['tas', 'tasmax', 'tasmin', 'pr'],
        historical_period: Tuple[int, int] = (1980, 2010),
        future_period: Tuple[int, int] = (2040, 2070),
        n_workers: Optional[int] = None,
        counties_subset: Optional[gpd.GeoDataFrame] = None,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Process counties in parallel (backward compatibility method).
        
        This method provides backward compatibility with the old API.
        
        Args:
            scenarios: List of climate scenarios
            variables: List of climate variables
            historical_period: (start_year, end_year) for historical
            future_period: (start_year, end_year) for future
            n_workers: Number of worker processes
            counties_subset: Subset of counties to process
            progress_callback: Progress callback function
            
        Returns:
            DataFrame with all results
        """
        logger.warning("Using deprecated process_parallel method. Please use process_all_counties instead.")
        
        # Update n_workers if specified
        if n_workers:
            self.n_workers = n_workers
        
        # Convert to new API
        indicators_config = self._create_default_indicators_config(variables)
        
        # Get county filter if subset provided
        counties_filter = None
        if counties_subset is not None:
            counties_filter = counties_subset['GEOID'].tolist()
        
        # Note: The new API doesn't use separate historical/future periods
        # This is handled internally based on scenarios
        logger.info(f"Historical period {historical_period} and future period {future_period} "
                   "parameters are ignored in unified processor")
        
        return self.process_all_counties(
            scenarios=scenarios,
            indicators_config=indicators_config,
            counties_filter=counties_filter,
            progress_callback=progress_callback
        )
    
    def save_results(
        self,
        results_df: pd.DataFrame,
        format: str = 'parquet'
    ) -> Path:
        """Save results to file.
        
        Args:
            results_df: DataFrame with results
            format: Output format ('parquet', 'csv', 'json')
            
        Returns:
            Path to saved file
        """
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'parquet':
            output_path = self.output_dir / f"climate_indicators_{timestamp}.parquet"
            results_df.to_parquet(output_path, index=False)
        elif format == 'csv':
            output_path = self.output_dir / f"climate_indicators_{timestamp}.csv"
            results_df.to_csv(output_path, index=False)
        elif format == 'json':
            output_path = self.output_dir / f"climate_indicators_{timestamp}.json"
            results_df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved results to {output_path}")
        return output_path
    
    # Phase 2: Tile-based batch processing methods
    
    def process_counties_by_tile(
        self,
        tile_id: str,
        tile_info: Dict[str, Any],
        scenarios: List[str],
        indicators_config: Dict[str, Dict[str, Any]],
        **kwargs
    ) -> Tuple[List[Dict], List[str]]:
        """Process multiple counties sharing the same spatial tile.
        
        This method loads data once for the entire tile and then extracts
        county-specific regions, significantly reducing I/O operations.
        
        Args:
            tile_id: Unique identifier for the tile
            tile_info: Dictionary with 'bounds' and 'counties' keys
            scenarios: List of climate scenarios to process
            indicators_config: Configuration for climate indicators
            **kwargs: Additional arguments passed to calculator
            
        Returns:
            Tuple of (results_list, failed_counties_list)
        """
        tile_bounds = tile_info['bounds']
        county_ids = tile_info['counties']
        
        logger.info(f"Processing tile {tile_id} with {len(county_ids)} counties")
        
        # Initialize calculator with optimizations enabled
        calculator = UnifiedClimateCalculator(
            base_data_path=str(self.base_data_path),
            merged_baseline_path=self.merged_baseline_path,
            cache_dir=kwargs.get('cache_dir'),
            use_zarr=kwargs.get('use_zarr', False),
            use_dask=kwargs.get('use_dask', True)
        )
        
        results = []
        failed_counties = []
        
        try:
            # Load data once for entire tile
            tile_data = {}
            for scenario in scenarios:
                scenario_data = {}
                
                # Load all variables for this scenario/tile
                for variable in ['tas', 'tasmax', 'tasmin', 'pr']:
                    # Check for Zarr store first
                    zarr_store = calculator.get_zarr_store(variable, scenario)
                    if zarr_store:
                        # Extract tile region from Zarr
                        scenario_data[variable] = zarr_store[variable].sel(
                            lat=slice(tile_bounds[1], tile_bounds[3]),
                            lon=slice(tile_bounds[0], tile_bounds[2])
                        )
                    else:
                        # Fall back to NetCDF loading
                        files = calculator._get_variable_scenario_files(variable, scenario)
                        if files:
                            ds = calculator.load_multiple_files(
                                files,
                                preselect_bounds={
                                    'lat': slice(tile_bounds[1], tile_bounds[3]),
                                    'lon': slice(tile_bounds[0], tile_bounds[2])
                                }
                            )
                            if ds and variable in ds:
                                scenario_data[variable] = ds[variable]
                
                tile_data[scenario] = scenario_data
            
            # Process each county using shared tile data
            for county_id in county_ids:
                try:
                    county_results = calculator.process_county_with_shared_data(
                        county_id, tile_data[scenarios[0]]  # Start with first scenario
                    )
                    
                    # Add county metadata
                    county_info = self.counties_gdf[self.counties_gdf['GEOID'] == county_id].iloc[0]
                    county_results.update({
                        'GEOID': county_id,
                        'NAME': county_info['NAME'],
                        'STATE': get_state_name(county_info.get('STATEFP', ''))
                    })
                    
                    results.append(county_results)
                    
                except Exception as e:
                    logger.error(f"Error processing county {county_id} in tile {tile_id}: {e}")
                    failed_counties.append(county_id)
                    
        except Exception as e:
            logger.error(f"Error loading data for tile {tile_id}: {e}")
            failed_counties.extend(county_ids)
        
        return results, failed_counties
    
    def create_spatial_tiles(self, tile_size_degrees: float = 2.0) -> Dict[str, Dict[str, Any]]:
        """Create spatial tiles for efficient batch processing.
        
        Args:
            tile_size_degrees: Size of each tile in degrees (default: 2.0)
            
        Returns:
            Dictionary of tile_id -> {'bounds': [...], 'counties': [...]}
        """
        from ..utils.shapefile_utils import CountyBoundsLookup
        
        lookup = CountyBoundsLookup(self.shapefile_path)
        return lookup.create_spatial_tiles(tile_size_degrees)
    
    def process_parallel_with_tiles(
        self,
        scenarios: List[str] = ['historical', 'ssp245'],
        indicators_config: Optional[Dict[str, Dict[str, Any]]] = None,
        tile_size_degrees: float = 2.0,
        max_counties_per_batch: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Process all counties using tile-based parallel processing.
        
        This method groups counties by spatial tiles to minimize data loading.
        
        Args:
            scenarios: List of climate scenarios
            indicators_config: Configuration for indicators
            tile_size_degrees: Size of spatial tiles in degrees
            max_counties_per_batch: Maximum counties per batch (None for auto)
            **kwargs: Additional arguments passed to calculator
            
        Returns:
            DataFrame with all results
        """
        # Create spatial tiles
        tiles = self.create_spatial_tiles(tile_size_degrees)
        logger.info(f"Created {len(tiles)} spatial tiles for processing")
        
        # Prepare tile batches
        tile_batches = list(tiles.items())
        
        # Process tiles in parallel
        all_results = []
        all_failed = []
        
        start_time = time.time()
        completed_counties = 0
        total_counties = len(self.counties_gdf)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit tile processing jobs
            future_to_tile = {
                executor.submit(
                    self.process_counties_by_tile,
                    tile_id,
                    tile_info,
                    scenarios,
                    indicators_config or self._create_default_indicators_config(['tas', 'tasmax', 'tasmin', 'pr']),
                    **kwargs
                ): (tile_id, len(tile_info['counties']))
                for tile_id, tile_info in tile_batches
            }
            
            # Process results as they complete
            for future in as_completed(future_to_tile):
                tile_id, n_counties = future_to_tile[future]
                
                try:
                    results, failed = future.result()
                    all_results.extend(results)
                    all_failed.extend(failed)
                    
                    completed_counties += n_counties
                    elapsed = time.time() - start_time
                    rate = completed_counties / elapsed if elapsed > 0 else 0
                    eta = (total_counties - completed_counties) / rate if rate > 0 else 0
                    
                    logger.info(
                        f"Completed tile {tile_id}: {len(results)} successful, {len(failed)} failed. "
                        f"Progress: {completed_counties}/{total_counties} ({completed_counties/total_counties*100:.1f}%). "
                        f"ETA: {eta/60:.1f} minutes"
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing tile {tile_id}: {e}")
        
        # Convert to DataFrame
        if all_results:
            results_df = pd.DataFrame(all_results)
        else:
            results_df = pd.DataFrame()
        
        # Log summary
        total_time = time.time() - start_time
        logger.info(f"\nProcessing complete:")
        logger.info(f"  Total time: {total_time/60:.1f} minutes")
        logger.info(f"  Counties processed: {len(all_results)}")
        logger.info(f"  Counties failed: {len(all_failed)}")
        logger.info(f"  Average time per county: {total_time/len(all_results):.1f}s" if all_results else "N/A")
        
        # Save failed counties list
        if all_failed:
            failed_path = self.output_dir / "failed_counties_tiles.json"
            save_json(all_failed, failed_path)
            logger.warning(f"Failed counties saved to {failed_path}")
        
        return results_df