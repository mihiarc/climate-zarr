#!/usr/bin/env python
"""Test script to verify climate data access."""

import os
from pathlib import Path

def check_climate_data():
    """Check if climate data is accessible and show structure."""
    base_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA")
    
    if not base_path.exists():
        print(f"❌ Base path does not exist: {base_path}")
        return
    
    print(f"✓ Base path exists: {base_path}")
    
    # Check for model directories
    print("\nAvailable climate models:")
    models = [d for d in base_path.iterdir() if d.is_dir()]
    for model in sorted(models):
        print(f"  - {model.name}")
    
    # Check NorESM2-LM specifically
    noresm_path = base_path / "NorESM2-LM"
    if noresm_path.exists():
        print(f"\n✓ NorESM2-LM model found")
        
        # Check for variables
        print("\nAvailable variables:")
        variables = [d for d in noresm_path.iterdir() if d.is_dir()]
        for var in sorted(variables):
            print(f"  - {var.name}")
            
            # Check scenarios for first variable
            if var == variables[0]:
                print(f"\n  Scenarios in {var.name}:")
                scenarios = [s for s in var.iterdir() if s.is_dir()]
                for scenario in sorted(scenarios):
                    print(f"    - {scenario.name}")
                    
                    # Count files
                    nc_files = list(scenario.glob("*.nc"))
                    print(f"      ({len(nc_files)} .nc files)")

if __name__ == "__main__":
    check_climate_data()