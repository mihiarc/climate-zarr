"""
Configuration for handling different US regions in climate data processing.

Different regions require special handling:
- Alaska: Crosses the dateline, extreme latitudes
- Hawaii: Pacific islands, tropical climate
- Territories: May have limited data availability
"""

from typing import Dict, Tuple, List


# Regional configurations
REGIONAL_CONFIG = {
    'conus': {
        'name': 'Continental United States',
        'state_codes': [str(i).zfill(2) for i in range(1, 57) if i not in [2, 15]],  # Exclude AK, HI
        'typical_temp_range': (-20, 40),  # Celsius
        'coordinate_system': 'standard',   # -180 to 180
        'data_availability': 'full'
    },
    'alaska': {
        'name': 'Alaska',
        'state_codes': ['02'],
        'typical_temp_range': (-40, 25),
        'coordinate_system': 'crosses_dateline',  # Special handling needed
        'data_availability': 'full',
        'notes': 'Some regions cross the international dateline'
    },
    'hawaii': {
        'name': 'Hawaii',
        'state_codes': ['15'],
        'typical_temp_range': (15, 32),
        'coordinate_system': 'standard',
        'data_availability': 'full'
    },
    'puerto_rico': {
        'name': 'Puerto Rico',
        'state_codes': ['72'],
        'typical_temp_range': (20, 35),
        'coordinate_system': 'standard',
        'data_availability': 'partial',
        'notes': 'Caribbean territory, 78 municipios'
    },
    'virgin_islands': {
        'name': 'US Virgin Islands',
        'state_codes': ['78'],
        'typical_temp_range': (22, 32),
        'coordinate_system': 'standard',
        'data_availability': 'limited',
        'notes': '3 main islands: St. Croix, St. Thomas, St. John'
    },
    'guam': {
        'name': 'Guam',
        'state_codes': ['66'],
        'typical_temp_range': (24, 32),
        'coordinate_system': 'western_pacific',  # Uses 0-360 longitude
        'data_availability': 'limited',
        'notes': 'Single county covering entire territory'
    },
    'american_samoa': {
        'name': 'American Samoa',
        'state_codes': ['60'],
        'typical_temp_range': (24, 30),
        'coordinate_system': 'crosses_dateline',
        'data_availability': 'very_limited'
    },
    'northern_marianas': {
        'name': 'Northern Mariana Islands',
        'state_codes': ['69'],
        'typical_temp_range': (24, 32),
        'coordinate_system': 'western_pacific',
        'data_availability': 'very_limited'
    }
}


def get_region_for_state(state_code: str) -> str:
    """Get region name for a given state code."""
    for region, config in REGIONAL_CONFIG.items():
        if state_code in config['state_codes']:
            return region
    return 'unknown'


def needs_dateline_handling(bounds: Tuple[float, float, float, float]) -> bool:
    """
    Check if a county's bounds require special dateline handling.
    
    Parameters
    ----------
    bounds : tuple
        (min_lon, min_lat, max_lon, max_lat)
        
    Returns
    -------
    bool
        True if special handling needed
    """
    min_lon, _, max_lon, _ = bounds
    
    # Check for dateline crossing
    if max_lon < min_lon:  # Wraps around
        return True
    
    # Check for extreme longitudes
    if abs(max_lon - min_lon) > 180:  # Spans more than half the globe
        return True
        
    return False


def adjust_bounds_for_dateline(bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Adjust bounds for counties that cross the dateline.
    
    Parameters
    ----------
    bounds : tuple
        Original bounds (min_lon, min_lat, max_lon, max_lat)
        
    Returns
    -------
    tuple
        Adjusted bounds
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # If max < min, it crosses the dateline
    if max_lon < min_lon:
        # Convert to 0-360 system
        if min_lon < 0:
            min_lon += 360
        if max_lon < 0:
            max_lon += 360
            
    return (min_lon, min_lat, max_lon, max_lat)


def get_climate_expectations(region: str) -> Dict:
    """
    Get expected climate characteristics for a region.
    
    Parameters
    ----------
    region : str
        Region name from REGIONAL_CONFIG
        
    Returns
    -------
    dict
        Expected climate characteristics
    """
    if region not in REGIONAL_CONFIG:
        return {}
        
    config = REGIONAL_CONFIG[region]
    temp_range = config['typical_temp_range']
    
    expectations = {
        'mean_temp_range': temp_range,
        'frost_days_range': (0, 0) if temp_range[0] > 0 else (30, 300),
        'hot_days_range': (0, 30) if temp_range[1] < 30 else (30, 200),
        'precipitation_type': 'tropical' if region in ['hawaii', 'guam', 'puerto_rico'] else 'varied'
    }
    
    return expectations


def validate_climate_data(df, region: str) -> List[str]:
    """
    Validate climate data makes sense for the region.
    
    Parameters
    ----------
    df : pd.DataFrame
        Climate data with standard columns
    region : str
        Region name
        
    Returns
    -------
    list
        List of validation warnings
    """
    warnings = []
    expectations = get_climate_expectations(region)
    
    if not expectations:
        return warnings
        
    # Check mean temperature
    mean_temp = df['tg_mean_C'].mean()
    expected_range = expectations['mean_temp_range']
    
    if mean_temp < expected_range[0] - 5:
        warnings.append(f"Mean temperature ({mean_temp:.1f}°C) is unusually low for {region}")
    elif mean_temp > expected_range[1] + 5:
        warnings.append(f"Mean temperature ({mean_temp:.1f}°C) is unusually high for {region}")
        
    # Check frost days
    if 'tn_days_below_32F' in df.columns:
        frost_days = df['tn_days_below_32F'].mean()
        expected_frost = expectations['frost_days_range']
        
        if region in ['hawaii', 'guam', 'puerto_rico', 'virgin_islands'] and frost_days > 0:
            warnings.append(f"Unexpected frost days ({frost_days:.1f}) in tropical region {region}")
            
    return warnings


# Test data availability patterns by region
DATA_AVAILABILITY_NOTES = {
    'full': 'Complete NEX-GDDP coverage expected',
    'partial': 'Some models/scenarios may be missing',
    'limited': 'Only basic scenarios available',
    'very_limited': 'May only have historical data or no data'
}


def get_recommended_processing_config(region: str) -> Dict:
    """
    Get recommended processing configuration for a region.
    
    Parameters
    ----------
    region : str
        Region name
        
    Returns
    -------
    dict
        Recommended configuration
    """
    if region not in REGIONAL_CONFIG:
        return {}
        
    config = REGIONAL_CONFIG[region]
    availability = config.get('data_availability', 'unknown')
    
    if availability == 'full':
        return {
            'scenarios': ['historical', 'ssp245', 'ssp370', 'ssp585'],
            'models': ['all'],
            'batch_size': 50
        }
    elif availability in ['partial', 'limited']:
        return {
            'scenarios': ['historical', 'ssp245'],
            'models': ['NorESM2-LM', 'GFDL-ESM4'],
            'batch_size': 20
        }
    else:  # very_limited
        return {
            'scenarios': ['historical'],
            'models': ['any_available'],
            'batch_size': 5
        }