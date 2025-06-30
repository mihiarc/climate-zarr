#!/usr/bin/env python3
"""
Quick test runner for regional coverage validation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from parallel_processor import ParallelClimateProcessor
from regional_config import REGIONAL_CONFIG, get_region_for_state


def check_regional_coverage(shapefile_path):
    """Check which regions are represented in the shapefile."""
    
    print("REGIONAL COVERAGE CHECK")
    print("="*60)
    
    processor = ParallelClimateProcessor(shapefile_path, "/dummy/path")
    
    # Count counties by region
    region_counts = {}
    total_counties = 0
    
    for _, county in processor.counties.iterrows():
        state_code = county['STATEFP']
        region = get_region_for_state(state_code)
        
        if region not in region_counts:
            region_counts[region] = {
                'count': 0,
                'examples': []
            }
        
        region_counts[region]['count'] += 1
        
        # Keep first 3 examples
        if len(region_counts[region]['examples']) < 3:
            region_counts[region]['examples'].append({
                'name': county['NAME'],
                'geoid': county['GEOID'],
                'bounds': county.geometry.bounds
            })
        
        total_counties += 1
    
    # Print results
    print(f"\nTotal counties/territories: {total_counties}")
    print("\nBreakdown by region:")
    
    for region in sorted(region_counts.keys()):
        if region == 'unknown':
            continue
            
        data = region_counts[region]
        config = REGIONAL_CONFIG.get(region, {})
        
        print(f"\n{region.upper()}")
        print(f"  Name: {config.get('name', 'Unknown')}")
        print(f"  Count: {data['count']}")
        print(f"  Data availability: {config.get('data_availability', 'unknown')}")
        
        print("  Examples:")
        for example in data['examples']:
            bounds = example['bounds']
            print(f"    {example['name']} ({example['geoid']})")
            print(f"      Bounds: {bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}")
    
    # Show unknown regions
    if 'unknown' in region_counts:
        print(f"\nUNKNOWN REGIONS: {region_counts['unknown']['count']}")
        for example in region_counts['unknown']['examples']:
            print(f"  {example['name']} ({example['geoid']}) - State: {example.get('state', 'N/A')}")


def validate_test_counties(shapefile_path):
    """Validate that our test counties exist in the shapefile."""
    
    print("\n" + "="*60)
    print("TEST COUNTY VALIDATION")
    print("="*60)
    
    processor = ParallelClimateProcessor(shapefile_path, "/dummy/path")
    
    # Test counties from conftest.py
    test_counties = {
        'conus': '08109',       # Saguache County, Colorado
        'alaska': '02220',      # Sitka, Alaska
        'hawaii': '15003',      # Honolulu, Hawaii
        'puerto_rico': '72115', # Quebradillas, Puerto Rico
        'virgin_islands': '78030', # St. Thomas, US Virgin Islands
        'guam': '66010'         # Guam
    }
    
    found_counties = 0
    
    for region, geoid in test_counties.items():
        county = processor.counties[processor.counties['GEOID'] == geoid]
        
        if len(county) > 0:
            county_info = county.iloc[0]
            bounds = county_info.geometry.bounds
            
            print(f"\n✓ {region.upper()}: {county_info['NAME']}")
            print(f"  GEOID: {geoid}")
            print(f"  State: {county_info['STATEFP']}")
            print(f"  Bounds: {bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, {bounds[3]:.3f}")
            found_counties += 1
        else:
            print(f"\n✗ {region.upper()}: County {geoid} not found")
    
    print(f"\nFound {found_counties}/{len(test_counties)} test counties")
    
    if found_counties == len(test_counties):
        print("✓ All test counties are available for testing")
    else:
        print("⚠ Some test counties are missing - tests may be skipped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check regional coverage in shapefile')
    parser.add_argument(
        '--shapefile',
        default='data/shapefiles/tl_2024_us_county.shp',
        help='Path to county shapefile'
    )
    
    args = parser.parse_args()
    
    shapefile_path = Path(args.shapefile)
    if not shapefile_path.exists():
        print(f"Error: Shapefile not found at {shapefile_path}")
        sys.exit(1)
    
    check_regional_coverage(str(shapefile_path))
    validate_test_counties(str(shapefile_path))