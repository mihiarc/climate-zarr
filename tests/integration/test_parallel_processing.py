"""
Integration tests for parallel climate data processing.
"""

import pytest
import pandas as pd
import time

from parallel_processor import ParallelClimateProcessor
from parallel_xclim_processor import ParallelXclimProcessor


class TestParallelProcessing:
    """Test suite for parallel processing functionality."""
    
    @pytest.mark.integration
    def test_basic_parallel_processing(self, shapefile_path, base_data_path, 
                                     test_counties, test_periods):
        """Test basic parallel processing with small dataset."""
        processor = ParallelClimateProcessor(shapefile_path, base_data_path)
        
        # Get test counties
        county_ids = test_counties['small']
        test_counties_df = processor.counties[
            processor.counties['GEOID'].isin(county_ids)
        ]
        
        # Process
        df = processor.process_parallel(
            counties_subset=test_counties_df,
            scenarios=['historical'],
            historical_period=test_periods['quick']['historical'],
            future_period=test_periods['quick']['future'],
            n_workers=2
        )
        
        # Validate results
        assert len(df) > 0
        assert len(df['GEOID'].unique()) == len(county_ids)
        assert 'tg_mean_C' in df.columns
        
    @pytest.mark.integration
    def test_backward_compatibility(self, shapefile_path, base_data_path, 
                                  test_counties, test_periods):
        """Test backward compatibility with ParallelXclimProcessor."""
        processor = ParallelXclimProcessor(shapefile_path, base_data_path)
        
        # Filter counties old-style
        processor.counties = processor.counties[
            processor.counties['GEOID'].isin(test_counties['small'])
        ].copy()
        
        # Process using old API
        df = processor.process_xclim_parallel(
            scenarios=['historical'],
            historical_period=test_periods['quick']['historical'],
            future_period=test_periods['quick']['future'],
            n_chunks=1
        )
        
        assert len(df) > 0
        assert 'tx90p_percent' in df.columns
        
    @pytest.mark.integration
    def test_data_consistency(self, shapefile_path, base_data_path, 
                            sample_county_info, test_periods):
        """Test that results are consistent across runs."""
        processor = ParallelClimateProcessor(shapefile_path, base_data_path)
        
        # Process same county twice
        county_ids = [sample_county_info['geoid']]
        test_counties_df = processor.counties[
            processor.counties['GEOID'].isin(county_ids)
        ]
        
        # First run
        df1 = processor.process_parallel(
            counties_subset=test_counties_df,
            scenarios=['historical'],
            historical_period=test_periods['quick']['historical'],
            future_period=test_periods['quick']['future'],
            n_workers=1
        )
        
        # Second run
        df2 = processor.process_parallel(
            counties_subset=test_counties_df,
            scenarios=['historical'],
            historical_period=test_periods['quick']['historical'],
            future_period=test_periods['quick']['future'],
            n_workers=1
        )
        
        # Compare results
        assert len(df1) == len(df2)
        
        # Check numeric columns are close (allowing for floating point differences)
        numeric_cols = ['tg_mean_C', 'tx_days_above_90F', 'tn_days_below_32F']
        for col in numeric_cols:
            if col in df1.columns:
                assert df1[col].subtract(df2[col]).abs().max() < 0.01
                
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multi_scenario_processing(self, shapefile_path, base_data_path,
                                     test_counties, test_periods, climate_scenarios):
        """Test processing multiple scenarios."""
        processor = ParallelClimateProcessor(shapefile_path, base_data_path)
        
        # Get test counties
        county_ids = test_counties['small']
        test_counties_df = processor.counties[
            processor.counties['GEOID'].isin(county_ids)
        ]
        
        # Process multiple scenarios
        df = processor.process_parallel(
            counties_subset=test_counties_df,
            scenarios=climate_scenarios['standard'],
            historical_period=test_periods['quick']['historical'],
            future_period=test_periods['quick']['future'],
            n_workers=2
        )
        
        # Validate
        assert len(df) > 0
        assert set(df['scenario'].unique()) == set(climate_scenarios['standard'])
        
        # Check that future scenarios have different values than historical
        hist_mean = df[df['scenario'] == 'historical']['tg_mean_C'].mean()
        future_mean = df[df['scenario'] == 'ssp245']['tg_mean_C'].mean()
        assert abs(future_mean - hist_mean) > 0.1  # Expect some warming
        
    @pytest.mark.integration
    def test_error_handling(self, shapefile_path, base_data_path):
        """Test error handling for invalid inputs."""
        processor = ParallelClimateProcessor(shapefile_path, base_data_path)
        
        # Test with non-existent counties
        fake_counties = processor.counties[
            processor.counties['GEOID'].isin(['99999'])
        ]
        
        # This should complete without crashing (empty results)
        df = processor.process_parallel(
            counties_subset=fake_counties,
            scenarios=['historical'],
            historical_period=(2009, 2010),
            future_period=(2040, 2041),
            n_workers=1
        )
        
        assert len(df) == 0 or df.empty