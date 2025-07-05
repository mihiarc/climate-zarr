#!/usr/bin/env python
"""Tests for utility modules."""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # Required for .rio accessor
from shapely.geometry import Polygon
from unittest.mock import patch

from climate_zarr.utils import (
    convert_units, 
    create_county_raster, 
    get_time_information
)
from climate_zarr.utils.data_utils import (
    convert_units,
    calculate_precipitation_stats,
    calculate_temperature_stats,
    calculate_tasmax_stats,
    calculate_tasmin_stats,
    calculate_statistics
)
from climate_zarr.utils.spatial_utils import (
    create_county_raster,
    get_time_information,
    get_coordinate_arrays,
    clip_county_data
)


@pytest.fixture
def sample_counties():
    """Create sample counties for testing."""
    counties = []
    for i in range(2):
        # Create small rectangular counties
        minx, miny = -100 + i*0.5, 40
        maxx, maxy = minx + 0.5, 41
        
        poly = Polygon([
            (minx, miny), (maxx, miny), 
            (maxx, maxy), (minx, maxy), (minx, miny)
        ])
        
        counties.append({
            'county_id': f'{i:05d}',
            'county_name': f'Test County {i}',
            'state': 'TX',
            'raster_id': i + 1,
            'geometry': poly
        })
    
    return gpd.GeoDataFrame(counties, crs='EPSG:4326')


@pytest.fixture
def sample_data():
    """Create sample xarray data."""
    time = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    lons = np.arange(-100.2, -99.3, 0.1)  # 10 points
    lats = np.arange(40.1, 40.9, 0.1)     # 8 points
    
    # Create realistic data
    np.random.seed(42)
    data = np.random.exponential(2e-6, size=(len(time), len(lats), len(lons)))
    
    da = xr.DataArray(
        data,
        coords={'time': time, 'lat': lats, 'lon': lons},
        dims=['time', 'lat', 'lon'],
        name='pr'
    )
    
    # Rename to spatial dimensions for rioxarray
    da = da.rename({'lat': 'y', 'lon': 'x'})
    return da.rio.write_crs('EPSG:4326')


class TestUnitConversion:
    """Test unit conversion utilities."""
    
    def test_precipitation_conversion(self):
        """Test precipitation unit conversion."""
        # kg/m²/s to mm/day
        data = np.array([1e-6, 2e-6, 5e-6])
        converted = convert_units(data, "kg/m2/s", "mm/day")
        expected = data * 86400  # seconds per day
        np.testing.assert_array_almost_equal(converted, expected)
    
    def test_temperature_conversion_kelvin_to_celsius(self):
        """Test temperature conversion from Kelvin to Celsius."""
        data = np.array([273.15, 283.15, 293.15])
        converted = convert_units(data, "K", "C")
        expected = np.array([0.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(converted, expected)
    
    def test_temperature_conversion_celsius_to_fahrenheit(self):
        """Test temperature conversion from Celsius to Fahrenheit."""
        data = np.array([0.0, 10.0, 20.0])
        converted = convert_units(data, "C", "F")
        expected = np.array([32.0, 50.0, 68.0])
        np.testing.assert_array_almost_equal(converted, expected)
    
    def test_unsupported_conversion(self):
        """Test unsupported unit conversion returns original data."""
        data = np.array([1.0, 2.0, 3.0])
        
        # Unsupported conversions should return original data
        converted = convert_units(data, "invalid_from", "invalid_to")
        np.testing.assert_array_equal(converted, data)
    
    def test_same_units(self):
        """Test conversion between same units."""
        data = np.array([1.0, 2.0, 3.0])
        converted = convert_units(data, "mm/day", "mm/day")
        np.testing.assert_array_equal(converted, data)
    
    def test_array_types(self):
        """Test conversion with different array types."""
        # Test with numpy array
        data_array = np.array([1e-6, 2e-6, 3e-6])
        converted = convert_units(data_array, "kg/m2/s", "mm/day")
        expected = np.array([0.0864, 0.1728, 0.2592])  # Correct conversion
        np.testing.assert_array_almost_equal(converted, expected)


class TestTimeInformation:
    """Test time information utilities."""
    
    def test_get_time_information(self, sample_data):
        """Test extracting time information from xarray data."""
        years, unique_years = get_time_information(sample_data)
        
        assert len(years) == len(sample_data.time)
        assert set(unique_years) == {2020}
        assert len(unique_years) == 1
        assert all(isinstance(year, (int, np.integer)) for year in years)
        assert all(isinstance(year, (int, np.integer)) for year in unique_years)
    
    def test_multi_year_data(self):
        """Test time information with multi-year data."""
        time = pd.date_range('2019-01-01', '2021-12-31', freq='D')
        data = xr.DataArray(
            np.random.rand(len(time), 5, 5),
            coords={'time': time, 'lat': np.arange(5), 'lon': np.arange(5)},
            dims=['time', 'lat', 'lon']
        )
        
        years, unique_years = get_time_information(data)
        
        assert len(years) == len(time)
        assert set(unique_years) == {2019, 2020, 2021}
        assert len(unique_years) == 3
    
    def test_monthly_data(self):
        """Test time information with monthly data."""
        time = pd.date_range('2020-01-01', '2020-12-31', freq='M')
        data = xr.DataArray(
            np.random.rand(len(time), 5, 5),
            coords={'time': time, 'lat': np.arange(5), 'lon': np.arange(5)},
            dims=['time', 'lat', 'lon']
        )
        
        years, unique_years = get_time_information(data)
        
        assert len(years) == len(time)
        assert set(unique_years) == {2020}
        assert all(year == 2020 for year in years)


class TestCountyRaster:
    """Test county raster creation utilities."""
    
    def test_create_county_raster(self, sample_counties):
        """Test creating county raster."""
        lats = np.arange(40.1, 40.9, 0.1)
        lons = np.arange(-100.2, -99.3, 0.1)
        
        county_raster = create_county_raster(sample_counties, lats, lons)
        
        assert county_raster.shape == (len(lats), len(lons))
        assert county_raster.dtype == np.uint16
        
        # Should have some non-zero values (counties)
        assert np.any(county_raster > 0)
        
        # Values should correspond to raster_id
        unique_values = np.unique(county_raster[county_raster > 0])
        expected_values = set(sample_counties['raster_id'].values)
        assert set(unique_values).issubset(expected_values)
    
    def test_create_county_raster_no_overlap(self, sample_counties):
        """Test creating county raster with no spatial overlap."""
        # Create grid that doesn't overlap with counties
        lats = np.arange(50.0, 51.0, 0.1)
        lons = np.arange(0.0, 1.0, 0.1)
        
        county_raster = create_county_raster(sample_counties, lats, lons)
        
        assert county_raster.shape == (len(lats), len(lons))
        assert county_raster.dtype == np.uint16
        
        # Should be all zeros (no overlap)
        assert np.all(county_raster == 0)
    
    def test_create_county_raster_empty_gdf(self):
        """Test creating county raster with empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(columns=['county_id', 'raster_id', 'geometry'])
        
        lats = np.arange(40.1, 40.9, 0.1)
        lons = np.arange(-100.2, -99.3, 0.1)
        
        county_raster = create_county_raster(empty_gdf, lats, lons)
        
        assert county_raster.shape == (len(lats), len(lons))
        assert county_raster.dtype == np.uint16
        assert np.all(county_raster == 0)


class TestDataStatistics:
    """Test data statistics calculation utilities."""
    
    def test_calculate_precipitation_stats(self):
        """Test precipitation statistics calculation."""
        # Create test data (mm/day)
        daily_values = np.array([0.0, 0.5, 10.0, 25.0, 50.0, 0.1])
        threshold_mm = 25.4
        county_info = {
            'county_id': '12345',
            'county_name': 'Test County',
            'state': 'TX'
        }
        
        stats = calculate_precipitation_stats(
            daily_values, threshold_mm, 2020, 'historical', county_info
        )
        
        assert stats['year'] == 2020
        assert stats['scenario'] == 'historical'
        assert stats['county_id'] == '12345'
        assert stats['county_name'] == 'Test County'
        assert stats['state'] == 'TX'
        
        # Check calculated values
        assert stats['total_annual_precip_mm'] == pytest.approx(85.6)
        assert stats['days_above_threshold'] == 1  # Only 50.0 > 25.4
        assert stats['mean_daily_precip_mm'] == pytest.approx(14.27, rel=1e-2)
        assert stats['max_daily_precip_mm'] == 50.0
        assert stats['dry_days'] == 1  # Only 0.0 < 0.1
    
    def test_calculate_temperature_stats(self):
        """Test temperature statistics calculation."""
        # Create test data (Celsius)
        daily_values = np.array([-10.0, -1.0, 0.0, 15.0, 25.0, 35.0])
        county_info = {
            'county_id': '12345',
            'county_name': 'Test County',
            'state': 'TX'
        }
        
        stats = calculate_temperature_stats(
            daily_values, 2020, 'historical', county_info
        )
        
        assert stats['year'] == 2020
        assert stats['scenario'] == 'historical'
        assert stats['county_id'] == '12345'
        
        # Check calculated values
        assert stats['mean_annual_temp_c'] == pytest.approx(10.67, rel=1e-2)
        assert stats['min_temp_c'] == -10.0
        assert stats['max_temp_c'] == 35.0
        assert stats['days_below_freezing'] == 2  # -10.0 and -1.0 < 0
        assert stats['days_above_30c'] == 1  # Only 35.0 > 30
        
        # Check growing degree days (base 10°C)
        expected_gdd = max(0, 15-10) + max(0, 25-10) + max(0, 35-10)  # 5 + 15 + 25 = 45
        assert stats['growing_degree_days'] == expected_gdd
    
    def test_calculate_tasmax_stats(self):
        """Test tasmax statistics calculation."""
        # Create test data (Celsius)
        daily_values = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
        county_info = {
            'county_id': '12345',
            'county_name': 'Test County',
            'state': 'TX'
        }
        
        stats = calculate_tasmax_stats(
            daily_values, 32.2, 2020, 'historical', county_info
        )
        
        assert stats['year'] == 2020
        assert stats['scenario'] == 'historical'
        assert stats['county_id'] == '12345'
        
        # Check calculated values
        assert stats['mean_annual_tasmax_c'] == pytest.approx(32.5)
        assert stats['min_tasmax_c'] == 20.0
        assert stats['max_tasmax_c'] == 45.0
        assert stats['days_above_35c'] == 2  # 40.0, 45.0 > 35 (35.0 is not > 35)
        assert stats['days_above_40c'] == 1  # Only 45.0 > 40 (40.0 is not > 40)
    
    def test_calculate_tasmin_stats(self):
        """Test tasmin statistics calculation."""
        # Create test data (Celsius)
        daily_values = np.array([-15.0, -5.0, 0.0, 5.0, 10.0, 15.0])
        county_info = {
            'county_id': '12345',
            'county_name': 'Test County',
            'state': 'TX'
        }
        
        stats = calculate_tasmin_stats(
            daily_values, 2020, 'historical', county_info
        )
        
        assert stats['year'] == 2020
        assert stats['scenario'] == 'historical'
        assert stats['county_id'] == '12345'
        
        # Check calculated values
        assert stats['mean_annual_tasmin_c'] == pytest.approx(1.67, rel=1e-2)
        assert stats['min_tasmin_c'] == -15.0
        assert stats['max_tasmin_c'] == 15.0
        assert stats['cold_days'] == 2  # -15.0, -5.0 < 0 (0.0 is not < 0)
        assert stats['extreme_cold_days'] == 1  # Only -15.0 <= -10
    
    def test_stats_with_missing_data(self):
        """Test statistics calculation with missing data."""
        # Create data with NaN values
        daily_values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        county_info = {
            'county_id': '12345',
            'county_name': 'Test County',
            'state': 'TX'
        }
        
        stats = calculate_precipitation_stats(
            daily_values, 25.4, 2020, 'historical', county_info
        )
        
        # Should handle NaN values gracefully
        assert not np.isnan(stats['total_annual_precip_mm'])
        assert not np.isnan(stats['mean_daily_precip_mm'])
        assert stats['total_annual_precip_mm'] == pytest.approx(9.0)  # 1 + 3 + 5
    
    def test_stats_with_all_zeros(self):
        """Test statistics calculation with all zero values."""
        daily_values = np.zeros(365)
        county_info = {
            'county_id': '12345',
            'county_name': 'Test County',
            'state': 'TX'
        }
        
        stats = calculate_precipitation_stats(
            daily_values, 25.4, 2020, 'historical', county_info
        )
        
        assert stats['total_annual_precip_mm'] == 0.0
        assert stats['mean_daily_precip_mm'] == 0.0
        assert stats['max_daily_precip_mm'] == 0.0
        assert stats['days_above_threshold'] == 0
        assert stats['dry_days'] == 365  # All days are dry


class TestSpatialUtils:
    """Test spatial processing utilities."""
    
    def test_clip_county_data(self, sample_counties, sample_data):
        """Test clipping data to county boundaries."""
        # Get first county geometry
        county_geometry = sample_counties.iloc[0].geometry
        
        clipped = clip_county_data(sample_data, county_geometry)
        
        assert isinstance(clipped, xr.DataArray)
        assert clipped.name == sample_data.name
        assert clipped.dims == sample_data.dims
        
        # Should have fewer spatial points than original
        assert len(clipped.y) <= len(sample_data.y)
        assert len(clipped.x) <= len(sample_data.x)
    
    def test_get_coordinate_arrays(self, sample_data):
        """Test getting coordinate arrays from data."""
        lats, lons = get_coordinate_arrays(sample_data)
        
        assert isinstance(lats, np.ndarray)
        assert isinstance(lons, np.ndarray)
        assert len(lats) == len(sample_data.y)
        assert len(lons) == len(sample_data.x)
        
        # Should match the original coordinates
        np.testing.assert_array_equal(lats, sample_data.y.values)
        np.testing.assert_array_equal(lons, sample_data.x.values)
    
    def test_calculate_statistics_integration(self, sample_counties, sample_data):
        """Test the general statistics calculation function."""
        # Get some sample data for first county
        county_info = {
            'county_id': sample_counties.iloc[0]['county_id'],
            'county_name': sample_counties.iloc[0]['county_name'],
            'state': sample_counties.iloc[0]['state']
        }
        
        # Create some daily precipitation data
        daily_precip = np.random.exponential(2, size=365)
        
        # Test precipitation statistics
        stats = calculate_statistics(
            daily_precip, 'pr', 25.4, 2020, 'test', county_info
        )
        
        assert isinstance(stats, dict)
        assert stats['variable'] == 'pr'
        assert stats['year'] == 2020
        assert stats['county_id'] == county_info['county_id']
        assert 'total_annual_precip_mm' in stats
    
    def test_spatial_utils_with_no_overlap(self, sample_counties):
        """Test spatial utilities with no spatial overlap."""
        # Create data that doesn't overlap with counties
        time = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        lons = np.array([0.0, 0.5, 1.0])  # Different location
        lats = np.array([0.0, 0.5, 1.0])  # Different location
        data = np.random.rand(len(time), len(lats), len(lons))
        
        da = xr.DataArray(
            data,
            coords={'time': time, 'lat': lats, 'lon': lons},
            dims=['time', 'lat', 'lon'],
            name='pr'
        )
        da = da.rename({'lat': 'y', 'lon': 'x'})
        da = da.rio.write_crs('EPSG:4326')
        
        # Get first county geometry
        county_geometry = sample_counties.iloc[0].geometry
        
        # Should raise NoDataInBounds when there's no overlap
        with pytest.raises(Exception):  # rioxarray.exceptions.NoDataInBounds
            clipped = clip_county_data(da, county_geometry)
        
        # Test coordinate extraction still works
        lats_out, lons_out = get_coordinate_arrays(da)
        np.testing.assert_array_equal(lats_out, lats)
        np.testing.assert_array_equal(lons_out, lons)


class TestUtilityErrorHandling:
    """Test error handling in utility functions."""
    
    def test_convert_units_with_invalid_input(self):
        """Test unit conversion with invalid input."""
        # String input should be handled gracefully or raise appropriate error
        try:
            result = convert_units("invalid", "K", "C")
            # If it doesn't raise an error, it should return the input unchanged
            assert result == "invalid"
        except (TypeError, ValueError):
            # This is also acceptable behavior
            pass
    
    def test_time_information_with_invalid_data(self):
        """Test time information with invalid data."""
        # Create data without time dimension
        data = xr.DataArray(
            np.random.rand(5, 5),
            coords={'lat': np.arange(5), 'lon': np.arange(5)},
            dims=['lat', 'lon']
        )
        
        with pytest.raises((KeyError, AttributeError)):
            get_time_information(data)
    
    def test_county_raster_with_invalid_coordinates(self):
        """Test county raster creation with invalid coordinates."""
        # Create GeoDataFrame with invalid geometry
        invalid_gdf = gpd.GeoDataFrame({
            'county_id': ['12345'],
            'raster_id': [1],
            'geometry': [None]  # Invalid geometry
        })
        
        lats = np.arange(40.1, 40.9, 0.1)
        lons = np.arange(-100.2, -99.3, 0.1)
        
        # Should handle invalid geometry gracefully
        county_raster = create_county_raster(invalid_gdf, lats, lons)
        
        assert county_raster.shape == (len(lats), len(lons))
        assert np.all(county_raster == 0)  # No valid geometries
    
    def test_stats_with_empty_data(self):
        """Test statistics calculation with empty data."""
        empty_data = np.array([])
        county_info = {
            'county_id': '12345',
            'county_name': 'Test County',
            'state': 'TX'
        }
        
        stats = calculate_precipitation_stats(
            empty_data, 25.4, 2020, 'historical', county_info
        )
        
        # Should handle empty data gracefully (may return None)
        if stats is None:
            # This is acceptable behavior for empty data
            assert True
        else:
            assert stats['total_annual_precip_mm'] == 0.0
            assert stats['mean_daily_precip_mm'] == 0.0
            assert stats['days_above_threshold'] == 0
            assert stats['dry_days'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 