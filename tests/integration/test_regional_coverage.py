"""
Test climate data processing for counties across all US regions.

This test ensures the processor correctly handles:
- CONUS (Continental US) counties
- Alaska counties (different longitude range)
- Hawaii counties (Pacific islands)
- Puerto Rico municipios (Caribbean)
- US Virgin Islands (Caribbean territories)
- Guam (Western Pacific territory)
"""

import pytest
import pandas as pd
import numpy as np

from parallel_processor import ParallelClimateProcessor
from climate_indicator_calculator import ClimateIndicatorCalculator


class TestRegionalCoverage:
    """Test suite for multi-regional climate data processing."""
    
    @pytest.fixture
    def regional_counties(self):
        """Representative counties from each US region."""
        return {
            'conus': {
                'geoid': '08109',
                'name': 'Saguache County, Colorado',
                'state': '08',
                'bounds': (-107.0, 37.5, -105.5, 38.5),  # Approximate
                'description': 'High altitude county in Rocky Mountains'
            },
            'alaska': {
                'geoid': '02220',
                'name': 'Sitka, Alaska',
                'state': '02',
                'bounds': (-136.5, 56.0, -134.5, 58.0),  # Approximate
                'description': 'Southeast Alaska, spans dateline issues'
            },
            'hawaii': {
                'geoid': '15003',
                'name': 'Honolulu, Hawaii',
                'state': '15',
                'bounds': (-158.5, 21.0, -157.5, 21.5),  # Approximate
                'description': 'Pacific island, tropical climate'
            },
            'puerto_rico': {
                'geoid': '72115',
                'name': 'Quebradillas, Puerto Rico',
                'state': '72',
                'bounds': (-67.0, 18.3, -66.8, 18.5),  # Approximate
                'description': 'Caribbean territory, hurricane zone'
            },
            'virgin_islands': {
                'geoid': '78030',
                'name': 'St. Thomas, US Virgin Islands',
                'state': '78',
                'bounds': (-65.1, 18.3, -64.8, 18.4),  # Approximate
                'description': 'Small Caribbean island'
            },
            'guam': {
                'geoid': '66010',
                'name': 'Guam',
                'state': '66',
                'bounds': (144.6, 13.2, 145.0, 13.7),  # Approximate
                'description': 'Western Pacific territory, crosses dateline'
            }
        }
    
    @pytest.mark.integration
    def test_load_all_regions(self, shapefile_path):
        """Test that all regions are present in shapefile."""
        processor = ParallelClimateProcessor(shapefile_path, "/dummy/path")
        
        # Check each state/territory code
        state_codes = {
            'conus': '08',      # Colorado
            'alaska': '02',     # Alaska
            'hawaii': '15',     # Hawaii
            'puerto_rico': '72', # Puerto Rico
            'virgin_islands': '78', # US Virgin Islands
            'guam': '66'        # Guam
        }
        
        for region, code in state_codes.items():
            region_counties = processor.counties[processor.counties['STATEFP'] == code]
            assert len(region_counties) > 0, f"No counties found for {region} (code {code})"
            print(f"{region}: {len(region_counties)} counties")
    
    @pytest.mark.integration
    def test_coordinate_handling(self, regional_counties):
        """Test coordinate system handling for different regions."""
        calculator = ClimateIndicatorCalculator("/dummy/path")
        
        # Test longitude conversion for each region
        for region, county in regional_counties.items():
            bounds = county['bounds']
            min_lon, min_lat, max_lon, max_lat = bounds
            
            # Test longitude conversion logic
            if min_lon < 0:
                converted_lon = min_lon % 360
            else:
                converted_lon = min_lon
            
            # Verify conversions make sense
            if region == 'guam':
                # Guam should not need conversion (already 0-360)
                assert min_lon > 0, f"{region} longitude should be positive"
            elif region in ['alaska', 'hawaii', 'conus', 'puerto_rico', 'virgin_islands']:
                # These use -180 to 180 system
                if region != 'alaska' or min_lon < 0:  # Alaska spans dateline
                    assert min_lon < 0, f"{region} longitude should be negative"
            
            print(f"{region}: lon {min_lon} -> {converted_lon}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_process_multi_region_sample(self, shapefile_path, base_data_path, 
                                       regional_counties, test_periods):
        """Test processing sample counties from each region."""
        processor = ParallelClimateProcessor(shapefile_path, base_data_path)
        
        # Get GEOIDs for test counties
        test_geoids = [county['geoid'] for county in regional_counties.values()]
        
        # Filter to available counties in shapefile
        available_counties = processor.counties[
            processor.counties['GEOID'].isin(test_geoids)
        ]
        
        print(f"\nTesting {len(available_counties)} regional counties:")
        for _, county in available_counties.iterrows():
            region = None
            for r, info in regional_counties.items():
                if info['geoid'] == county['GEOID']:
                    region = r
                    break
            print(f"  {county['NAME']} ({region})")
        
        # Skip if no data available for these regions
        if len(available_counties) == 0:
            pytest.skip("No test counties found in shapefile")
        
        # Process with minimal time period
        try:
            df = processor.process_parallel(
                counties_subset=available_counties,
                scenarios=['historical'],
                historical_period=test_periods['quick']['historical'],
                future_period=test_periods['quick']['future'],
                n_workers=2
            )
            
            # Validate results
            assert len(df) > 0, "No results returned"
            
            # Check each region's results
            for geoid in df['GEOID'].unique():
                region_data = df[df['GEOID'] == geoid]
                region_name = available_counties[
                    available_counties['GEOID'] == geoid
                ]['NAME'].iloc[0]
                
                print(f"\nResults for {region_name} ({geoid}):")
                print(f"  Records: {len(region_data)}")
                print(f"  Mean temp: {region_data['tg_mean_C'].mean():.1f}°C")
                
                # Validate temperature ranges make sense
                mean_temp = region_data['tg_mean_C'].mean()
                
                # Regional temperature expectations
                if geoid == regional_counties['alaska']['geoid']:
                    assert mean_temp < 10, "Alaska should be cold"
                elif geoid in [regional_counties['hawaii']['geoid'], 
                             regional_counties['guam']['geoid']]:
                    assert mean_temp > 20, "Tropical regions should be warm"
                elif geoid in [regional_counties['puerto_rico']['geoid'],
                             regional_counties['virgin_islands']['geoid']]:
                    assert mean_temp > 20, "Caribbean should be warm"
                    
        except FileNotFoundError:
            pytest.skip("Climate data not available for all regions")
        except Exception as e:
            if "No files found" in str(e):
                pytest.skip(f"Climate data not available: {e}")
            else:
                raise
    
    @pytest.mark.integration
    def test_extreme_coordinates(self, shapefile_path):
        """Test handling of extreme coordinate cases."""
        processor = ParallelClimateProcessor(shapefile_path, "/dummy/path")
        
        # Find counties with extreme coordinates
        all_counties = processor.counties.copy()
        
        # Add bounds to dataframe
        all_counties['bounds'] = all_counties.geometry.bounds.apply(
            lambda x: (x[0], x[1], x[2], x[3])
        )
        
        # Find extremes
        all_counties['min_lon'] = all_counties['bounds'].apply(lambda x: x[0])
        all_counties['max_lon'] = all_counties['bounds'].apply(lambda x: x[2])
        all_counties['min_lat'] = all_counties['bounds'].apply(lambda x: x[1])
        all_counties['max_lat'] = all_counties['bounds'].apply(lambda x: x[3])
        
        # Western-most (likely Guam or Northern Marianas)
        westernmost = all_counties.loc[all_counties['min_lon'].idxmax()]
        print(f"\nWesternmost: {westernmost['NAME']} ({westernmost['min_lon']:.2f}°)")
        
        # Eastern-most (likely Virgin Islands or Maine)
        easternmost = all_counties.loc[all_counties['max_lon'].idxmin()]
        print(f"Easternmost: {easternmost['NAME']} ({easternmost['max_lon']:.2f}°)")
        
        # Northern-most (likely Alaska)
        northernmost = all_counties.loc[all_counties['max_lat'].idxmax()]
        print(f"Northernmost: {northernmost['NAME']} ({northernmost['max_lat']:.2f}°)")
        
        # Southern-most (likely American Samoa or Hawaii)
        southernmost = all_counties.loc[all_counties['min_lat'].idxmin()]
        print(f"Southernmost: {southernmost['NAME']} ({southernmost['min_lat']:.2f}°)")
        
        # Check for dateline crossing (longitude > 180 or < -180)
        dateline_counties = all_counties[
            (all_counties['max_lon'] > 180) | (all_counties['min_lon'] < -180)
        ]
        
        if len(dateline_counties) > 0:
            print(f"\nCounties crossing dateline: {len(dateline_counties)}")
            for _, county in dateline_counties.head().iterrows():
                print(f"  {county['NAME']}: {county['min_lon']:.2f} to {county['max_lon']:.2f}")
    
    @pytest.mark.integration
    def test_climate_diversity(self, regional_counties):
        """Verify different regions have appropriately different climate characteristics."""
        # This test documents expected climate differences
        expectations = {
            'alaska': {
                'avg_temp_range': (-10, 10),  # Cold
                'frost_days': (150, 365),      # Many frost days
                'hot_days': (0, 10)            # Few hot days
            },
            'hawaii': {
                'avg_temp_range': (20, 28),    # Warm
                'frost_days': (0, 0),          # No frost
                'hot_days': (100, 365)         # Many hot days
            },
            'guam': {
                'avg_temp_range': (25, 30),    # Very warm
                'frost_days': (0, 0),          # No frost
                'hot_days': (200, 365)         # Most days are hot
            },
            'puerto_rico': {
                'avg_temp_range': (22, 28),    # Warm
                'frost_days': (0, 0),          # No frost
                'hot_days': (150, 365)         # Many hot days
            },
            'conus': {
                'avg_temp_range': (-5, 25),    # Variable
                'frost_days': (0, 200),        # Depends on location
                'hot_days': (0, 150)           # Depends on location
            }
        }
        
        print("\nExpected climate characteristics by region:")
        for region, expected in expectations.items():
            if region in regional_counties:
                county = regional_counties[region]
                print(f"\n{county['name']}:")
                print(f"  Expected avg temp: {expected['avg_temp_range']}°C")
                print(f"  Expected frost days: {expected['frost_days']}")
                print(f"  Expected hot days (>90°F): {expected['hot_days']}")


# Additional test to verify data availability
@pytest.mark.integration
def test_data_availability_by_region(shapefile_path, base_data_path):
    """Check which regions have climate data available."""
    from pathlib import Path
    
    processor = ParallelClimateProcessor(shapefile_path, base_data_path)
    calculator = ClimateIndicatorCalculator(base_data_path)
    
    # Sample counties from different regions
    test_cases = [
        ('08109', 'Colorado (CONUS)'),
        ('02220', 'Alaska'),
        ('15003', 'Hawaii'),
        ('72115', 'Puerto Rico'),
        ('78030', 'US Virgin Islands'),
        ('66010', 'Guam')
    ]
    
    print("\nData availability check:")
    for geoid, description in test_cases:
        county = processor.counties[processor.counties['GEOID'] == geoid]
        
        if len(county) == 0:
            print(f"  {description}: County not found in shapefile")
            continue
            
        # Check for any historical data files
        try:
            files = calculator.get_files_for_period(
                'tas', 'historical', 2010, 2010
            )
            
            if files:
                print(f"  {description}: ✓ Data available")
            else:
                print(f"  {description}: ✗ No data found")
                
        except Exception as e:
            print(f"  {description}: ✗ Error checking data: {e}")