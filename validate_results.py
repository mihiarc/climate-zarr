#!/usr/bin/env python3
"""Validation script to verify climate indicator calculations match business logic."""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.unified_calculator import UnifiedClimateCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def validate_single_county():
    """Process a single county to validate results."""
    
    # Configuration
    base_data_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    
    # Initialize calculator
    logger.info("Initializing calculator...")
    calculator = UnifiedClimateCalculator(
        base_data_path=base_data_path,
        baseline_period=(1980, 2010),
        use_dask=False  # Disable for debugging
    )
    
    # Test with Los Angeles County
    county_info = {
        'geoid': '06037',
        'name': 'Los Angeles',
        'state': 'California'
    }
    
    # LA County bounds (approximate)
    county_bounds = [-118.9517, 33.7037, -117.6462, 34.8233]
    
    logger.info(f"\nProcessing {county_info['name']} County ({county_info['geoid']})")
    logger.info(f"Bounds: {county_bounds}")
    
    # Define indicators matching business requirements
    indicators_config = {
        # Temperature extremes
        'tx90p': {
            'xclim_func': 'tx90p',
            'variable': 'tasmax',
            'freq': 'YS',
            'description': 'Days exceeding 90th percentile of baseline max temperature'
        },
        'tn10p': {
            'xclim_func': 'tn10p', 
            'variable': 'tasmin',
            'freq': 'YS',
            'description': 'Nights below 10th percentile of baseline min temperature'
        },
        'tx_days_above_90F': {
            'xclim_func': 'tx_days_above',
            'variable': 'tasmax',
            'thresh': '305.37 K',  # 90°F = 32.2°C = 305.37 K
            'freq': 'YS',
            'description': 'Days with max temperature above 90°F'
        },
        'tn_days_below_32F': {
            'xclim_func': 'tn_days_below',
            'variable': 'tasmin', 
            'thresh': '273.15 K',  # 32°F = 0°C = 273.15 K
            'freq': 'YS',
            'description': 'Nights with min temperature below 32°F (frost days)'
        },
        'tg_mean': {
            'xclim_func': 'tg_mean',
            'variable': 'tas',
            'freq': 'YS',
            'description': 'Annual mean temperature'
        },
        'precip_accumulation': {
            'xclim_func': 'precip_accumulation',
            'variable': 'pr',
            'freq': 'YS',
            'description': 'Total annual precipitation'
        }
    }
    
    # Process historical data first (2005-2010 for validation)
    logger.info("\n1. Processing historical data (2005-2010)...")
    
    try:
        results = calculator.calculate_indicators(
            scenarios=['historical'],
            county_bounds=county_bounds,
            county_info=county_info,
            indicators_config=indicators_config,
            year_range=(2005, 2010)
        )
        
        if results:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(results)
            
            logger.info(f"\n✓ Successfully processed {len(df)} records")
            
            # Display raw results
            logger.info("\nRaw results:")
            print(df.to_string())
            
            # Validate data quality
            logger.info("\n2. Data Quality Validation:")
            
            # Check temperature values
            if 'tg_mean' in df.columns:
                mean_temps = df['tg_mean'].values
                logger.info(f"\nMean temperatures (°C):")
                for year, temp in zip(df['year'].values, mean_temps):
                    logger.info(f"  {year}: {temp:.2f}°C ({temp * 9/5 + 32:.2f}°F)")
                
                # LA should have mean temps around 17-19°C (62-66°F)
                avg_temp = mean_temps.mean()
                logger.info(f"\nAverage: {avg_temp:.2f}°C ({avg_temp * 9/5 + 32:.2f}°F)")
                if 15 <= avg_temp <= 22:
                    logger.info("✓ Temperature range looks reasonable for LA County")
                else:
                    logger.warning("⚠ Temperature range seems off for LA County")
            
            # Check hot days
            if 'tx_days_above_90F' in df.columns:
                hot_days = df['tx_days_above_90F'].values
                logger.info(f"\nDays above 90°F:")
                for year, days in zip(df['year'].values, hot_days):
                    logger.info(f"  {year}: {days} days")
                
                avg_hot_days = hot_days.mean()
                logger.info(f"\nAverage: {avg_hot_days:.1f} days/year")
                # LA typically has 20-40 days above 90°F
                if 10 <= avg_hot_days <= 60:
                    logger.info("✓ Hot days count looks reasonable")
                else:
                    logger.warning("⚠ Hot days count seems off")
            
            # Check frost days
            if 'tn_days_below_32F' in df.columns:
                frost_days = df['tn_days_below_32F'].values
                logger.info(f"\nFrost days (below 32°F):")
                for year, days in zip(df['year'].values, frost_days):
                    logger.info(f"  {year}: {days} days")
                
                avg_frost_days = frost_days.mean()
                logger.info(f"\nAverage: {avg_frost_days:.1f} days/year")
                # LA should have very few frost days (0-5)
                if avg_frost_days <= 10:
                    logger.info("✓ Frost days count looks reasonable for LA")
                else:
                    logger.warning("⚠ Too many frost days for LA County")
            
            # Check precipitation
            if 'precip_accumulation' in df.columns:
                precip = df['precip_accumulation'].values
                logger.info(f"\nAnnual precipitation (mm):")
                for year, mm in zip(df['year'].values, precip):
                    inches = mm / 25.4
                    logger.info(f"  {year}: {mm:.1f} mm ({inches:.1f} inches)")
                
                avg_precip = precip.mean()
                avg_inches = avg_precip / 25.4
                logger.info(f"\nAverage: {avg_precip:.1f} mm ({avg_inches:.1f} inches/year)")
                # LA typically gets 12-15 inches of rain
                if 8 <= avg_inches <= 25:
                    logger.info("✓ Precipitation looks reasonable for LA")
                else:
                    logger.warning("⚠ Precipitation seems off for LA County")
            
            # Check percentile indicators
            if 'tx90p' in df.columns and 'tn10p' in df.columns:
                logger.info(f"\nPercentile indicators:")
                logger.info("  tx90p (hot days exceeding baseline 90th percentile):")
                for year, pct in zip(df['year'].values, df['tx90p'].values):
                    logger.info(f"    {year}: {pct:.1f}%")
                
                logger.info("  tn10p (cold nights below baseline 10th percentile):")  
                for year, pct in zip(df['year'].values, df['tn10p'].values):
                    logger.info(f"    {year}: {pct:.1f}%")
                
                # These should average around 10% for the baseline period
                avg_tx90p = df['tx90p'].mean()
                avg_tn10p = df['tn10p'].mean()
                logger.info(f"\n  Average tx90p: {avg_tx90p:.1f}%")
                logger.info(f"  Average tn10p: {avg_tn10p:.1f}%")
                
                if 5 <= avg_tx90p <= 15 and 5 <= avg_tn10p <= 15:
                    logger.info("✓ Percentile indicators look reasonable")
                else:
                    logger.warning("⚠ Percentile indicators may need checking")
            
            # Save results
            output_file = Path("results/validation/la_county_historical.csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"\n✓ Results saved to: {output_file}")
            
            # Also process one future year for comparison
            logger.info("\n3. Processing future scenario (ssp245, year 2050)...")
            future_results = calculator.calculate_indicators(
                scenarios=['ssp245'],
                county_bounds=county_bounds,
                county_info=county_info,
                indicators_config=indicators_config,
                year_range=(2050, 2050)
            )
            
            if future_results:
                future_df = pd.DataFrame(future_results)
                logger.info("\nFuture scenario (2050):")
                print(future_df.to_string())
                
                # Compare with historical
                if 'tg_mean' in future_df.columns and 'tg_mean' in df.columns:
                    hist_avg_temp = df['tg_mean'].mean()
                    future_temp = future_df['tg_mean'].iloc[0]
                    warming = future_temp - hist_avg_temp
                    logger.info(f"\nProjected warming by 2050: {warming:.2f}°C ({warming * 9/5:.2f}°F)")
                
                future_output = Path("results/validation/la_county_future.csv")
                future_df.to_csv(future_output, index=False)
                logger.info(f"✓ Future results saved to: {future_output}")
            
        else:
            logger.error("No results returned!")
            
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_single_county()