"""
Pytest configuration and shared fixtures for climate data processing tests.
"""

import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
import os


@pytest.fixture
def base_data_path():
    """Path to climate data - can be overridden with environment variable."""
    return os.environ.get('CLIMATE_DATA_PATH', '/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM')


@pytest.fixture
def shapefile_path():
    """Path to county shapefile."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/shapefiles/tl_2024_us_county.shp'))


@pytest.fixture
def test_counties():
    """Standard set of test counties for consistency."""
    return {
        'small': ['31039', '53069'],  # 2 counties for quick tests
        'medium': ['31039', '53069', '48453', '06037', '17031'],  # 5 counties
        'large': ['31039', '53069', '48453', '06037', '17031', 
                  '36061', '12086', '04013', '39049', '27053']  # 10 counties
    }


@pytest.fixture
def test_periods():
    """Standard test periods."""
    return {
        'quick': {
            'historical': (2009, 2010),  # 2 years
            'future': (2040, 2041)
        },
        'standard': {
            'historical': (2005, 2010),  # 6 years
            'future': (2040, 2045)
        },
        'full': {
            'historical': (1980, 2010),  # 31 years
            'future': (2040, 2070)
        }
    }


@pytest.fixture
def climate_variables():
    """Standard climate variables."""
    return ['tas', 'tasmax', 'tasmin', 'pr']


@pytest.fixture
def climate_scenarios():
    """Standard climate scenarios."""
    return {
        'minimal': ['historical'],
        'standard': ['historical', 'ssp245'],
        'full': ['historical', 'ssp245', 'ssp370', 'ssp585']
    }


@pytest.fixture
def sample_county_info():
    """Sample county information for unit tests."""
    return {
        'geoid': '31039',
        'name': 'Cuming',
        'state': '31',
        'bounds': (-96.7887, 41.7193, -96.1251, 42.2088)
    }


@pytest.fixture
def regional_test_counties():
    """Counties from different US regions for comprehensive testing."""
    return {
        'conus': {
            'geoid': '08109',
            'name': 'Saguache County',
            'state': '08',
            'description': 'High altitude Colorado county'
        },
        'alaska': {
            'geoid': '02220', 
            'name': 'Sitka',
            'state': '02',
            'description': 'Southeast Alaska'
        },
        'hawaii': {
            'geoid': '15003',
            'name': 'Honolulu',
            'state': '15',
            'description': 'Main Hawaiian island'
        },
        'puerto_rico': {
            'geoid': '72115',
            'name': 'Quebradillas',
            'state': '72',
            'description': 'Caribbean territory'
        },
        'virgin_islands': {
            'geoid': '78030',
            'name': 'St. Thomas',
            'state': '78',
            'description': 'Caribbean island territory'
        },
        'guam': {
            'geoid': '66010',
            'name': 'Guam',
            'state': '66',
            'description': 'Western Pacific territory'
        }
    }


@pytest.fixture
def expected_indicators():
    """Expected climate indicators in results."""
    return [
        'tx90p_percent',
        'tx_days_above_90F',
        'tn10p_percent',
        'tn_days_below_32F',
        'tg_mean_C',
        'days_precip_over_25.4mm',
        'precip_accumulation_mm'
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)