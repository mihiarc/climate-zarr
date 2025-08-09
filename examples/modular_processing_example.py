#!/usr/bin/env python
"""
Example: Using the New Modular Climate Zarr Architecture

This example demonstrates how to use the refactored modular architecture
for processing climate data at the county level.
"""

from pathlib import Path
from climate_zarr import ModernCountyProcessor
from climate_zarr.processors import PrecipitationProcessor, TemperatureProcessor
from climate_zarr.processors.processing_strategies import UltraFastStrategy, VectorizedStrategy
from climate_zarr.utils import convert_units, create_county_raster
from climate_zarr.utils.data_utils import calculate_precipitation_stats

def example_basic_usage():
    """Example 1: Basic usage - same as before, no changes needed!"""
    print("🔄 Example 1: Basic Usage (Backward Compatible)")
    
    # This works exactly the same as the old monolithic version
    with ModernCountyProcessor(n_workers=4) as processor:
        # Load shapefile  
        gdf = processor.prepare_shapefile("tl_2024_us_county/tl_2024_us_county.shp")
        
        # Process precipitation data
        results = processor.process_zarr_data(
            zarr_path="data/climate_data.zarr",
            gdf=gdf,
            variable="pr",
            scenario="ssp370", 
            threshold=25.4,
            chunk_by_county=True
        )
        
        print(f"✅ Processed {len(results)} records")
        return results

def example_advanced_usage():
    """Example 2: Advanced usage with specific processors"""
    print("\n🚀 Example 2: Advanced Usage with Specific Processors")
    
    # Use a specific processor directly
    processor = PrecipitationProcessor(n_workers=8)
    
    # Load shapefile
    gdf = processor.prepare_shapefile("tl_2024_us_county/tl_2024_us_county.shp")
    
    # Process with custom settings
    results = processor.process_zarr_file(
        zarr_path="data/climate_data.zarr",
        gdf=gdf,
        scenario="ssp370",
        threshold_mm=50.0,  # Custom threshold
        chunk_by_county=False  # Use vectorized processing
    )
    
    print(f"✅ Processed {len(results)} records with custom settings")
    return results

def example_custom_strategy():
    """Example 3: Using custom processing strategies"""
    print("\n⚡ Example 3: Custom Processing Strategies")
    
    # Create processors
    temp_processor = TemperatureProcessor(n_workers=4)
    gdf = temp_processor.prepare_shapefile("tl_2024_us_county/tl_2024_us_county.shp")
    
    # Load data manually
    import xarray as xr
    ds = xr.open_zarr("data/temperature_data.zarr")
    tas_data = ds['tas']
    
    # Use ultra-fast strategy for large datasets
    strategy = UltraFastStrategy()
    results = strategy.process(
        data=tas_data,
        gdf=gdf,
        variable='tas',
        scenario='historical',
        threshold=0.0,  # Not used for temperature
        n_workers=4
    )
    
    print(f"⚡ Ultra-fast processing completed: {len(results)} records")
    return results

def example_utilities():
    """Example 4: Using individual utilities"""
    print("\n🛠️ Example 4: Using Individual Utilities")
    
    import numpy as np
    import pandas as pd
    
    # Unit conversion
    temp_kelvin = np.array([273.15, 283.15, 293.15])
    temp_celsius = convert_units(temp_kelvin, "K", "C")
    print(f"Converted {temp_kelvin} K to {temp_celsius} °C")
    
    # Calculate precipitation statistics directly
    daily_precip = np.random.exponential(2.0, 365)  # Simulate daily precip
    county_info = {
        'county_id': '12345',
        'county_name': 'Example County',
        'state': 'EX'
    }
    
    stats = calculate_precipitation_stats(
        daily_values=daily_precip,
        threshold_mm=25.4,
        year=2020,
        scenario='historical',
        county_info=county_info
    )
    
    print(f"📊 Calculated stats: {stats['total_annual_precip_mm']:.1f} mm annual precipitation")
    return stats

def example_multiple_variables():
    """Example 5: Processing multiple variables efficiently"""
    print("\n🌡️ Example 5: Processing Multiple Variables")
    
    # Process multiple variables with the same counties
    with ModernCountyProcessor(n_workers=6) as processor:
        # Load shapefile once
        gdf = processor.prepare_shapefile("tl_2024_us_county/tl_2024_us_county.shp")
        
        results = {}
        
        # Process precipitation
        print("  Processing precipitation...")
        results['precipitation'] = processor.process_zarr_data(
            zarr_path="data/precip_data.zarr",
            gdf=gdf,
            variable="pr",
            scenario="ssp370",
            threshold=25.4
        )
        
        # Process temperature  
        print("  Processing temperature...")
        results['temperature'] = processor.process_zarr_data(
            zarr_path="data/temp_data.zarr", 
            gdf=gdf,
            variable="tas",
            scenario="ssp370"
        )
        
        # Process daily max temperature
        print("  Processing daily max temperature...")
        results['tasmax'] = processor.process_zarr_data(
            zarr_path="data/tasmax_data.zarr",
            gdf=gdf, 
            variable="tasmax",
            scenario="ssp370",
            threshold=32.2  # 90°F in Celsius
        )
        
        print(f"✅ Processed {len(results)} variables")
        return results

def example_extending_for_new_variable():
    """Example 6: Extending the architecture for a new variable"""
    print("\n🔧 Example 6: Extending for New Variables")
    
    from climate_zarr.processors.base_processor import BaseCountyProcessor
    from climate_zarr.processors.processing_strategies import VectorizedStrategy
    
    class HumidityProcessor(BaseCountyProcessor):
        """Example processor for humidity data."""
        
        def process_variable_data(self, data, gdf, scenario, **kwargs):
            # Convert units if needed (humidity is usually %)
            humidity_data = self._standardize_coordinates(data)
            
            # Choose processing strategy
            strategy = VectorizedStrategy()
            
            # Process the data
            return strategy.process(
                data=humidity_data,
                gdf=gdf,
                variable='humidity',
                scenario=scenario,
                threshold=kwargs.get('humidity_threshold', 80.0),
                n_workers=self.n_workers
            )
    
    # Use the new processor
    humidity_processor = HumidityProcessor(n_workers=4)
    print("✅ Created custom HumidityProcessor")
    
    # This shows how easy it is to extend the architecture
    return humidity_processor

def main():
    """Run all examples."""
    print("🏗️ Climate Zarr Modular Architecture Examples")
    print("=" * 50)
    
    # Note: These examples assume you have the data files
    # In practice, you'd uncomment and run the ones you need
    
    try:
        # Example 1: Basic usage (commented out - needs real data)
        # example_basic_usage()
        
        # Example 2: Advanced usage (commented out - needs real data)  
        # example_advanced_usage()
        
        # Example 3: Custom strategies (commented out - needs real data)
        # example_custom_strategy()
        
        # Example 4: Utilities (this one works without data files)
        example_utilities()
        
        # Example 5: Multiple variables (commented out - needs real data)
        # example_multiple_variables()
        
        # Example 6: Extending architecture (this one works)
        example_extending_for_new_variable()
        
        print("\n🎉 Examples completed successfully!")
        print("\nTo run the data processing examples, make sure you have:")
        print("- County shapefile: tl_2024_us_county/tl_2024_us_county.shp")
        print("- Zarr datasets: data/*.zarr")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("This is expected if you don't have the data files.")

if __name__ == "__main__":
    main() 