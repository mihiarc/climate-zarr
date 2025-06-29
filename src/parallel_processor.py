#!/usr/bin/env python3
"""
Parallel processor for climate data using the core climate indicator calculator.
This module handles all parallel processing logic separately from climate calculations.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Callable
import time
from climate_indicator_calculator import ClimateIndicatorCalculator


class ParallelClimateProcessor:
    """
    Handles parallel processing of climate data across multiple counties.
    
    This class manages:
    - Loading and preparing county data
    - Distributing work across multiple processes
    - Progress tracking and error handling
    - Result aggregation
    """
    
    def __init__(self, 
                 counties_shapefile_path: str,
                 base_data_path: str,
                 baseline_period: Tuple[int, int] = (1980, 2010)):
        """
        Initialize the parallel processor.
        
        Parameters
        ----------
        counties_shapefile_path : str
            Path to counties shapefile
        base_data_path : str
            Base path to climate data
        baseline_period : tuple
            (start_year, end_year) for baseline calculations
        """
        self.shapefile_path = Path(counties_shapefile_path)
        self.base_data_path = Path(base_data_path)
        self.baseline_period = baseline_period
        
        # Load counties
        self._load_counties()
        
    def _load_counties(self):
        """Load and prepare county shapefile data."""
        print(f"Loading counties from {self.shapefile_path}")
        self.counties = gpd.read_file(self.shapefile_path)
        
        # Ensure we have required fields
        required_fields = ['GEOID', 'NAME', 'STATEFP']
        missing = [f for f in required_fields if f not in self.counties.columns]
        if missing:
            raise ValueError(f"Missing required fields in shapefile: {missing}")
            
        print(f"Loaded {len(self.counties)} counties")
        
    def prepare_county_info(self, county_row) -> Dict:
        """
        Prepare county information for processing.
        
        Parameters
        ----------
        county_row : pd.Series
            Row from counties GeoDataFrame
            
        Returns
        -------
        dict
            County information including bounds
        """
        bounds = county_row.geometry.bounds
        
        return {
            'geoid': county_row['GEOID'],
            'name': county_row['NAME'],
            'state': county_row.get('STATEFP', 'Unknown'),
            'bounds': bounds  # (minx, miny, maxx, maxy)
        }
    
    @staticmethod
    def process_county_batch(batch_info: Dict) -> List[Dict]:
        """
        Static method to process a batch of counties.
        
        This method is designed to be called by worker processes.
        
        Parameters
        ----------
        batch_info : dict
            Dictionary containing:
            - counties: List of county info dictionaries
            - base_data_path: Path to climate data
            - baseline_period: Baseline period tuple
            - scenarios: List of scenarios
            - variables: List of variables
            - historical_period: Historical period tuple
            - future_period: Future period tuple
            - batch_id: Batch identifier
            
        Returns
        -------
        list
            Results for all counties in the batch
        """
        # Create calculator instance in worker process
        calculator = ClimateIndicatorCalculator(
            base_data_path=batch_info['base_data_path'],
            baseline_period=batch_info['baseline_period']
        )
        
        batch_results = []
        batch_id = batch_info['batch_id']
        counties = batch_info['counties']
        
        print(f"Worker processing batch {batch_id} with {len(counties)} counties")
        
        for i, county_info in enumerate(counties):
            try:
                # Process single county
                county_results = calculator.process_county(
                    county_info=county_info,
                    scenarios=batch_info['scenarios'],
                    variables=batch_info['variables'],
                    historical_period=batch_info['historical_period'],
                    future_period=batch_info['future_period']
                )
                
                batch_results.extend(county_results)
                
                if (i + 1) % 10 == 0:
                    print(f"  Batch {batch_id}: Processed {i + 1}/{len(counties)} counties")
                    
            except Exception as e:
                print(f"  ERROR in batch {batch_id}, county {county_info['name']}: {str(e)}")
                # Continue processing other counties
                continue
                
        print(f"Batch {batch_id} complete: {len(batch_results)} records")
        return batch_results
    
    def create_batches(self, 
                      counties_subset: Optional[pd.DataFrame] = None,
                      batch_size: Optional[int] = None) -> List[List[Dict]]:
        """
        Create batches of counties for processing.
        
        Parameters
        ----------
        counties_subset : pd.DataFrame, optional
            Subset of counties to process. If None, uses all counties.
        batch_size : int, optional
            Size of each batch. If None, automatically determined.
            
        Returns
        -------
        list
            List of batches, where each batch is a list of county info dicts
        """
        counties_to_process = counties_subset if counties_subset is not None else self.counties
        
        # Prepare county info
        county_infos = [
            self.prepare_county_info(row) 
            for _, row in counties_to_process.iterrows()
        ]
        
        # Determine batch size
        if batch_size is None:
            n_workers = min(mp.cpu_count(), 16)
            batch_size = max(1, len(county_infos) // (n_workers * 4))  # 4 batches per worker
            
        # Create batches
        batches = []
        for i in range(0, len(county_infos), batch_size):
            batches.append(county_infos[i:i + batch_size])
            
        return batches
    
    def process_parallel(self,
                        scenarios: List[str] = ['historical', 'ssp245'],
                        variables: List[str] = ['tas', 'tasmax', 'tasmin', 'pr'],
                        historical_period: Tuple[int, int] = (1980, 2010),
                        future_period: Tuple[int, int] = (2040, 2070),
                        n_workers: Optional[int] = None,
                        counties_subset: Optional[pd.DataFrame] = None,
                        progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Process counties in parallel.
        
        Parameters
        ----------
        scenarios : list
            Climate scenarios to process
        variables : list
            Climate variables to process
        historical_period : tuple
            (start_year, end_year) for historical analysis
        future_period : tuple
            (start_year, end_year) for future projections
        n_workers : int, optional
            Number of worker processes. If None, uses CPU count.
        counties_subset : pd.DataFrame, optional
            Subset of counties to process
        progress_callback : callable, optional
            Function to call with progress updates
            
        Returns
        -------
        pd.DataFrame
            Results dataframe
        """
        if n_workers is None:
            n_workers = min(mp.cpu_count(), 16)
            
        # Create batches
        batches = self.create_batches(counties_subset)
        n_counties = sum(len(batch) for batch in batches)
        
        print(f"\nParallel processing configuration:")
        print(f"  Total counties: {n_counties}")
        print(f"  Number of batches: {len(batches)}")
        print(f"  Batch size: ~{len(batches[0]) if batches else 0} counties")
        print(f"  Worker processes: {n_workers}")
        print(f"  Scenarios: {', '.join(scenarios)}")
        print(f"  Historical period: {historical_period[0]}-{historical_period[1]}")
        print(f"  Future period: {future_period[0]}-{future_period[1]}")
        
        # Prepare batch info for workers
        batch_infos = []
        for i, batch in enumerate(batches):
            batch_info = {
                'counties': batch,
                'base_data_path': str(self.base_data_path),
                'baseline_period': self.baseline_period,
                'scenarios': scenarios,
                'variables': variables,
                'historical_period': historical_period,
                'future_period': future_period,
                'batch_id': i
            }
            batch_infos.append(batch_info)
            
        # Process batches in parallel
        all_results = []
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_county_batch, batch_info): i 
                for i, batch_info in enumerate(batch_infos)
            }
            
            # Process completed batches
            completed = 0
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    completed += 1
                    
                    # Progress update
                    progress = (completed / len(batches)) * 100
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(batches) - completed) / rate if rate > 0 else 0
                    
                    print(f"\nProgress: {completed}/{len(batches)} batches ({progress:.1f}%)")
                    print(f"  Elapsed: {elapsed:.1f}s, Rate: {rate:.2f} batches/s, ETA: {eta:.1f}s")
                    
                    if progress_callback:
                        progress_callback(completed, len(batches), elapsed)
                        
                except Exception as e:
                    print(f"\nERROR in batch {batch_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
        # Summary
        total_time = time.time() - start_time
        print(f"\nProcessing complete:")
        print(f"  Total time: {total_time:.1f} seconds")
        print(f"  Counties processed: {n_counties}")
        print(f"  Records generated: {len(all_results)}")
        print(f"  Average time per county: {total_time/n_counties:.2f} seconds")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Sort by county, scenario, and year
        if not df.empty:
            df = df.sort_values(['GEOID', 'scenario', 'year'])
            
        return df
    
    def process_test_counties(self,
                            test_geoids: List[str],
                            **kwargs) -> pd.DataFrame:
        """
        Process a small set of test counties.
        
        Parameters
        ----------
        test_geoids : list
            List of GEOIDs to process
        **kwargs
            Additional arguments passed to process_parallel
            
        Returns
        -------
        pd.DataFrame
            Results for test counties
        """
        # Filter to test counties
        test_counties = self.counties[self.counties['GEOID'].isin(test_geoids)].copy()
        
        if len(test_counties) == 0:
            raise ValueError(f"No counties found with GEOIDs: {test_geoids}")
            
        print(f"\nTesting with {len(test_counties)} counties:")
        for _, county in test_counties.iterrows():
            print(f"  - {county['NAME']}, {county['STATEFP']} (GEOID: {county['GEOID']})")
            
        # Process with single worker for testing
        kwargs['n_workers'] = kwargs.get('n_workers', 1)
        
        return self.process_parallel(counties_subset=test_counties, **kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = ParallelClimateProcessor(
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010)
    )
    
    # Test with a few counties
    test_results = processor.process_test_counties(
        test_geoids=['31039', '53069'],  # Cuming NE, Wahkiakum WA
        scenarios=['historical', 'ssp245'],
        historical_period=(2005, 2010),
        future_period=(2040, 2045)
    )
    
    # Save test results
    test_results.to_csv("parallel_processor_test.csv", index=False)
    print(f"\nTest results saved: {len(test_results)} records")