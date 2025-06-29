"""
Unit tests for climate indicator calculator.
"""

import pytest
import numpy as np
import xarray as xr
import pandas as pd
from unittest.mock import Mock, patch

from climate_indicator_calculator import ClimateIndicatorCalculator


class TestClimateIndicatorCalculator:
    """Test suite for climate indicator calculations."""
    
    @pytest.fixture
    def calculator(self, base_data_path):
        """Create calculator instance."""
        return ClimateIndicatorCalculator(
            base_data_path=base_data_path,
            baseline_period=(1980, 2010)
        )
    
    @pytest.fixture
    def mock_data(self):
        """Create mock climate data for testing."""
        # Create time series
        dates = pd.date_range('2010-01-01', '2010-12-31', freq='D')
        n_days = len(dates)
        
        # Create mock temperature data (varying around 280K)
        tas = 280 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.randn(n_days)
        tasmax = tas + 5 + np.random.randn(n_days) * 0.5
        tasmin = tas - 5 + np.random.randn(n_days) * 0.5
        
        # Create mock precipitation (mm/day equivalent)
        pr = np.abs(np.random.randn(n_days) * 0.0001)  # kg/m2/s
        
        return {
            'time': dates,
            'tas': tas,
            'tasmax': tasmax,
            'tasmin': tasmin,
            'pr': pr
        }
    
    def test_calculate_indicators(self, calculator, mock_data):
        """Test indicator calculation with mock data."""
        # Create mock thresholds
        thresholds = {
            'tasmax_p90_doy': xr.DataArray(
                np.full(365, 295),  # 90th percentile threshold
                dims=['dayofyear'],
                coords={'dayofyear': np.arange(1, 366)}
            ),
            'tasmin_p10_doy': xr.DataArray(
                np.full(365, 270),  # 10th percentile threshold
                dims=['dayofyear'],
                coords={'dayofyear': np.arange(1, 366)}
            )
        }
        
        # Add units
        thresholds['tasmax_p90_doy'].attrs['units'] = 'K'
        thresholds['tasmin_p10_doy'].attrs['units'] = 'K'
        
        # Calculate indicators
        indicators = calculator.calculate_indicators(mock_data, thresholds)
        
        # Validate results
        assert 'tx90p' in indicators
        assert 'tn10p' in indicators
        assert 'tx_days_above_90F' in indicators
        assert 'tn_days_below_32F' in indicators
        assert 'tg_mean' in indicators
        assert 'days_precip_over_25.4mm' in indicators
        assert 'precip_accumulation' in indicators
        
        # Check that all indicators have yearly frequency
        for ind_name, ind_data in indicators.items():
            assert len(ind_data) == 1  # One value for the year
            
    def test_get_files_for_period(self, calculator):
        """Test file filtering by period."""
        # This test would need actual files or mocking
        # For now, just test the method exists and handles empty results
        files = calculator.get_files_for_period(
            variable='tas',
            scenario='historical',
            start_year=2100,  # Future year that won't exist
            end_year=2101
        )
        
        assert isinstance(files, list)
        assert len(files) == 0  # No files for future years
        
    def test_extract_county_data_error_handling(self, calculator):
        """Test error handling in data extraction."""
        # Test with empty file list
        with pytest.raises(ValueError, match="No files provided"):
            calculator.extract_county_data(
                files=[],
                variable='tas',
                bounds=(-100, 40, -99, 41)
            )
            
    @patch('xarray.open_mfdataset')
    def test_extract_county_data_coordinate_conversion(self, mock_open, calculator):
        """Test longitude coordinate conversion."""
        # Create mock dataset with 0-360 longitude
        mock_ds = Mock()
        mock_var = Mock()
        mock_ds.__getitem__.return_value = mock_var
        mock_var.sel.return_value = mock_var
        mock_var.weighted.return_value = mock_var
        mock_var.mean.return_value = mock_var
        mock_var.compute.return_value = xr.DataArray(
            np.random.randn(365),
            dims=['time'],
            coords={'time': pd.date_range('2010-01-01', periods=365)}
        )
        mock_var.lat = xr.DataArray([40, 41], dims=['lat'])
        mock_open.return_value = mock_ds
        
        # Test with negative longitude (should be converted)
        result = calculator.extract_county_data(
            files=['/fake/file.nc'],
            variable='tas',
            bounds=(-100, 40, -99, 41)  # Negative longitude
        )
        
        # Check that sel was called with converted longitude
        calls = mock_var.sel.call_args_list
        lon_slice = calls[0][1]['lon']
        assert lon_slice.start > 0  # Should be converted to 0-360
        
    def test_baseline_period_validation(self):
        """Test baseline period is properly set."""
        calculator = ClimateIndicatorCalculator(
            base_data_path='/fake/path',
            baseline_period=(1990, 2020)
        )
        
        assert calculator.baseline_period == (1990, 2020)
        
    def test_process_county_integration(self, calculator, sample_county_info):
        """Test full county processing (would need real data or extensive mocking)."""
        # This is more of an integration test placeholder
        # In a real test suite, you'd mock the file operations
        
        # For now, just verify the method exists and accepts correct parameters
        assert hasattr(calculator, 'process_county')
        
        # Test would fail with real call due to missing data
        # results = calculator.process_county(
        #     county_info=sample_county_info,
        #     scenarios=['historical'],
        #     variables=['tas', 'tasmax', 'tasmin', 'pr'],
        #     historical_period=(2009, 2010),
        #     future_period=(2040, 2041)
        # )