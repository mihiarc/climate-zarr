"""
Performance tests for optimization validation.
"""

import pytest
import time
import pandas as pd

from parallel_processor import ParallelClimateProcessor
from optimized_parallel_processor import OptimizedParallelProcessor


class TestOptimizationPerformance:
    """Test suite for performance optimizations."""
    
    @pytest.mark.performance
    def test_caching_speedup(self, shapefile_path, base_data_path, 
                           sample_county_info, test_periods):
        """Test that caching provides expected speedup."""
        from optimized_climate_calculator import OptimizedClimateCalculator
        
        calculator = OptimizedClimateCalculator(
            base_data_path=base_data_path,
            enable_caching=True
        )
        
        # First run - no cache
        start = time.time()
        thresholds1 = calculator.calculate_baseline_percentiles(
            sample_county_info['bounds']
        )
        time_no_cache = time.time() - start
        
        # Second run - with cache
        start = time.time()
        thresholds2 = calculator.calculate_baseline_percentiles(
            sample_county_info['bounds']
        )
        time_with_cache = time.time() - start
        
        # Validate
        assert time_with_cache < time_no_cache * 0.5  # At least 2x speedup
        assert 'tasmax_p90_doy' in thresholds1
        assert 'tasmax_p90_doy' in thresholds2
        
        # Clear cache
        calculator.clear_cache(memory_only=True)
        
    @pytest.mark.performance
    def test_optimized_vs_original(self, shapefile_path, base_data_path,
                                  test_counties, test_periods):
        """Compare optimized vs original processor performance."""
        # Get test counties
        original_processor = ParallelClimateProcessor(shapefile_path, base_data_path)
        county_ids = test_counties['small']
        test_counties_df = original_processor.counties[
            original_processor.counties['GEOID'].isin(county_ids)
        ]
        
        # Test original
        start = time.time()
        df_orig = original_processor.process_parallel(
            counties_subset=test_counties_df,
            scenarios=['historical'],
            historical_period=test_periods['quick']['historical'],
            future_period=test_periods['quick']['future'],
            n_workers=2
        )
        time_original = time.time() - start
        
        # Test optimized (first run)
        optimized_processor = OptimizedParallelProcessor(
            shapefile_path, base_data_path, enable_caching=True
        )
        
        start = time.time()
        df_opt = optimized_processor.process_parallel_optimized(
            counties_subset=test_counties_df,
            scenarios=['historical'],
            historical_period=test_periods['quick']['historical'],
            future_period=test_periods['quick']['future'],
            n_workers=2,
            counties_per_batch=2
        )
        time_optimized = time.time() - start
        
        # Validate results match
        assert len(df_orig) == len(df_opt)
        assert time_optimized <= time_original * 1.5  # Allow some overhead on first run
        
    @pytest.mark.performance
    @pytest.mark.slow
    def test_scaling_efficiency(self, shapefile_path, base_data_path,
                              test_counties, test_periods):
        """Test parallel scaling efficiency."""
        processor = OptimizedParallelProcessor(
            shapefile_path, base_data_path, enable_caching=True
        )
        
        county_ids = test_counties['medium']  # 5 counties
        test_counties_df = processor.counties[
            processor.counties['GEOID'].isin(county_ids)
        ]
        
        results = {}
        
        # Test with different worker counts
        for n_workers in [1, 2, 4]:
            start = time.time()
            df = processor.process_parallel_optimized(
                counties_subset=test_counties_df,
                scenarios=['historical'],
                historical_period=test_periods['quick']['historical'],
                future_period=test_periods['quick']['future'],
                n_workers=n_workers,
                counties_per_batch=2
            )
            elapsed = time.time() - start
            
            results[n_workers] = {
                'time': elapsed,
                'records': len(df)
            }
        
        # Check scaling efficiency
        speedup_2 = results[1]['time'] / results[2]['time']
        speedup_4 = results[1]['time'] / results[4]['time']
        
        # Should see reasonable scaling (>1.5x for 2 workers, >2.5x for 4)
        assert speedup_2 > 1.5
        assert speedup_4 > 2.5
        
    @pytest.mark.performance
    def test_batch_size_impact(self, shapefile_path, base_data_path,
                              test_counties, test_periods):
        """Test impact of different batch sizes."""
        processor = OptimizedParallelProcessor(
            shapefile_path, base_data_path, enable_caching=True
        )
        
        county_ids = test_counties['medium']
        test_counties_df = processor.counties[
            processor.counties['GEOID'].isin(county_ids)
        ]
        
        results = {}
        
        # Test different batch sizes
        for batch_size in [1, 3, 5]:
            start = time.time()
            df = processor.process_parallel_optimized(
                counties_subset=test_counties_df,
                scenarios=['historical'],
                historical_period=test_periods['quick']['historical'],
                future_period=test_periods['quick']['future'],
                n_workers=2,
                counties_per_batch=batch_size
            )
            elapsed = time.time() - start
            
            results[batch_size] = elapsed
        
        # Larger batches should generally be faster (less overhead)
        assert results[5] <= results[1] * 1.1  # Allow 10% variance
        
    @pytest.mark.performance
    def test_memory_usage(self, shapefile_path, base_data_path,
                         test_counties, test_periods):
        """Test memory usage stays reasonable."""
        import psutil
        import os
        
        processor = OptimizedParallelProcessor(
            shapefile_path, base_data_path, enable_caching=True
        )
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple counties
        county_ids = test_counties['medium']
        test_counties_df = processor.counties[
            processor.counties['GEOID'].isin(county_ids)
        ]
        
        df = processor.process_parallel_optimized(
            counties_subset=test_counties_df,
            scenarios=['historical'],
            historical_period=test_periods['quick']['historical'],
            future_period=test_periods['quick']['future'],
            n_workers=2,
            counties_per_batch=3
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for 5 counties)
        assert memory_increase < 500
        assert len(df) > 0