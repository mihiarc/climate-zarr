#!/usr/bin/env python
"""
output_management_example.py

Demonstrates the use of the standardized output management system in the
climate-zarr package. This script shows how to configure output directories,
apply different naming conventions, generate standardized output paths, save
data with metadata, and list existing outputs.

Sections:
    1. Basic Output Configuration
    2. Creating Output Manager
    3. Generating Standardized Paths
    4. Testing Different Naming Conventions
    5. Saving Data with Metadata
    6. Environment Variable Configuration
    7. Programmatic Configuration
    8. Listing Existing Outputs

Requirements:
    - climate-zarr package
    - pandas

Usage:
    python output_management_example.py

Author: Your Name
Date: 2025-08-04
"""

from pathlib import Path
import pandas as pd
from climate_zarr.climate_config import get_config
from climate_zarr.utils.output_utils import get_output_manager, standardize_output_path


def main():
    """
    Demonstrate output management features for climate data processing.
    Shows configuration, path generation, saving with metadata, and listing outputs.
    """
    
    print("🎯 Climate Data Output Management Example")
    print("=" * 50)
    
    # 1. Basic Configuration
    print("\n📁 1. Basic Output Configuration")
    config = get_config()
    print(f"   Base directory: {config.output.base_output_dir}")
    print(f"   Naming convention: {config.output.naming_convention}")
    print(f"   Organize by variable: {config.output.organize_by_variable}")
    print(f"   Organize by region: {config.output.organize_by_region}")
    
    # 2. Output Manager
    print("\n🔧 2. Creating Output Manager")
    manager = get_output_manager()
    print("   ✅ Output manager created")
    
    # 3. Generate Standardized Paths
    print("\n📄 3. Generating Standardized Paths")
    # Example scenarios for output path generation
    scenarios = [
        ("pr", "conus", "historical", 25.4),
        ("tasmax", "alaska", "ssp370", 35.0),
        ("tasmin", "hawaii", "ssp245", 0.0),
        ("tas", "puerto_rico", "historical", None)
    ]
    
    for variable, region, scenario, threshold in scenarios:
        # Generate and display standardized output path for each scenario
        path = manager.get_output_path(
            variable=variable,
            region=region,
            scenario=scenario,
            threshold=threshold
        )
        print(f"   {variable:>6} | {region:>11} | {scenario:>10} → {path.name}")
    
    # 4. Test Different Naming Conventions
    print("\n🎨 4. Different Naming Conventions")
    conventions = ["simple", "descriptive", "detailed", "iso"]
    for convention in conventions:
        # Change naming convention and show resulting file name
        config.output.naming_convention = convention
        manager_conv = get_output_manager(config)
        path = manager_conv.get_output_path(
            variable="pr",
            region="conus", 
            scenario="historical",
            threshold=25.4
        )
        print(f"   {convention:>11}: {path.name}")
    
    # 5. Save Sample Data with Metadata
    print("\n💾 5. Saving Data with Metadata")
    # Create sample DataFrame for demonstration
    sample_data = pd.DataFrame({
        'county_id': ['001', '002', '003'],
        'county_name': ['County A', 'County B', 'County C'],
        'year': [2020, 2020, 2020],
        'total_annual_precip_mm': [1200.5, 980.2, 1450.8],
        'days_above_threshold': [45, 32, 67]
    })
    
    # Save data and metadata using standardized output path
    output_path = manager.get_output_path(
        variable="pr",
        region="example",
        scenario="demo",
        threshold=25.4
    )
    metadata = {
        "data_source": "Example dataset",
        "processing_date": "2025-08-04",
        "notes": "Demonstration of output management system"
    }
    saved_path = manager.save_with_metadata(
        data=sample_data,
        output_path=output_path,
        metadata=metadata
    )
    print(f"   ✅ Saved data to: {saved_path}")
    print(f"   📋 Metadata saved to: {saved_path.with_suffix('.metadata.json')}")
    
    # 6. Environment Variable Configuration
    print("\n🌍 6. Environment Variable Configuration")
    print("   You can configure output settings using environment variables:")
    print("   export CLIMATE_BASE_OUTPUT_DIR='./my_results'")
    print("   export CLIMATE_NAMING_CONVENTION='iso'")
    print("   export CLIMATE_CREATE_SUBDIRS='false'")
    
    # 7. Programmatic Configuration
    print("\n⚙️ 7. Programmatic Configuration")
    print("   Or configure directly in Python:")
    print("""
   from climate_zarr.climate_config import get_config
   
   config = get_config()
   config.output.base_output_dir = Path('./custom_outputs')
   config.output.naming_convention = 'detailed'
   config.output.include_timestamp = True
   config.setup_directories()
   """)
    
    # 8. List Existing Outputs
    print("\n📋 8. Listing Existing Outputs")
    outputs = manager.list_outputs(variable="pr", region="example")
    if outputs:
        print(f"   Found {len(outputs)} files for pr/example:")
        for output in outputs:
            if output.is_file():
                print(f"     📄 {output.name}")
    else:
        print("   No existing outputs found")
    
    print("\n🎉 Output Management Example Complete!")
    print("\nKey Benefits:")
    print("✅ Consistent file naming across all tools")
    print("✅ Organized directory structure")
    print("✅ Automatic metadata generation")
    print("✅ Configurable naming conventions")
    print("✅ Environment variable support")
    print("✅ Easy integration with existing code")


if __name__ == "__main__":
    main()