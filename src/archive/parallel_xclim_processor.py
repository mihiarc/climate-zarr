#!/usr/bin/env python3
"""
Parallel processor for NEX-GDDP climate data with xclim indicators.
This is a wrapper around the modular parallel processor for backward compatibility.
"""

import pandas as pd
from typing import Optional, Tuple, List
from parallel_processor import ParallelClimateProcessor


class ParallelXclimProcessor:
    """
    Backward-compatible wrapper around the new modular parallel processor.
    
    This class maintains the same API as the original processor while using
    the new modular design underneath.
    """
    
    def __init__(self, 
                 counties_shapefile_path: str, 
                 base_data_path: str, 
                 baseline_period: Tuple[int, int] = (1980, 2010)):
        """
        Initialize the parallel xclim processor.
        
        Parameters
        ----------
        counties_shapefile_path : str
            Path to the US counties shapefile
        base_data_path : str
            Base path to NEX-GDDP climate data
        baseline_period : tuple, optional
            (start_year, end_year) for calculating percentile thresholds.
            Default is (1980, 2010) - standard 30-year climatological period.
        """
        # Use the new modular processor
        self.processor = ParallelClimateProcessor(
            counties_shapefile_path=counties_shapefile_path,
            base_data_path=base_data_path,
            baseline_period=baseline_period
        )
        
        # Expose counties for backward compatibility
        self.counties = self.processor.counties
        self.baseline_period = baseline_period
        
        print(f"Initialized with {len(self.counties)} counties")
        print(f"Baseline period for percentiles: {baseline_period[0]}-{baseline_period[1]}")
    
    def process_xclim_parallel(self, 
                              scenarios: List[str] = ['historical', 'ssp245'], 
                              variables: List[str] = ['tas', 'tasmax', 'tasmin', 'pr'],
                              historical_period: Tuple[int, int] = (1980, 2010),
                              future_period: Tuple[int, int] = (2040, 2070),
                              n_chunks: Optional[int] = None) -> pd.DataFrame:
        """
        Process all counties in parallel to calculate xclim indicators.
        
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
        n_chunks : int, optional
            Number of parallel chunks. If None, uses min(cpu_count, 16)
            
        Returns
        -------
        pd.DataFrame
            Results with annual indicator values for each county
        """
        # Map n_chunks to n_workers for new API
        n_workers = n_chunks
        
        # Check if we have a subset of counties (for backward compatibility)
        counties_subset = None
        if len(self.counties) != len(self.processor.counties):
            counties_subset = self.counties
        
        # Call the new processor
        return self.processor.process_parallel(
            scenarios=scenarios,
            variables=variables,
            historical_period=historical_period,
            future_period=future_period,
            n_workers=n_workers,
            counties_subset=counties_subset
        )


# Import the new base parallel processor to maintain backward compatibility
# This allows existing imports to continue working
try:
    import sys
    sys.path.append('../archive')
    from parallel_nex_gddp_processor import ParallelNEXGDDP_CountyProcessor
except ImportError:
    # If the archive import fails, define a minimal base class
    class ParallelNEXGDDP_CountyProcessor:
        """Minimal base class for compatibility."""
        pass


# Example usage
if __name__ == "__main__":
    # Initialize processor with standard 30-year baseline
    processor = ParallelXclimProcessor(
        counties_shapefile_path="../data/shapefiles/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010)
    )
    
    # Test with a subset of counties
    processor.counties = processor.counties[
        processor.counties['GEOID'].isin(['31039', '53069'])
    ].copy()
    
    # Process climate indicators
    df = processor.process_xclim_parallel(
        scenarios=['historical', 'ssp245'],
        variables=['tas', 'tasmax', 'tasmin', 'pr'],
        historical_period=(2005, 2010),
        future_period=(2040, 2045),
        n_chunks=1
    )
    
    # Save results
    df.to_csv("climate_indicators_modular.csv", index=False)
    print(f"\nSaved {len(df)} records to climate_indicators_modular.csv")