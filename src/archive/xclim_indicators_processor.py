#!/usr/bin/env python3
"""
Calculate xclim climate indicators for NEX-GDDP county-level data
"""

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xclim
from xclim import atmos
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the base processor
import importlib.util
spec = importlib.util.spec_from_file_location("nex_gddp_county_processor", 
                                              "nex-gddp-county-processor.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LocalNEXGDDP_CountyProcessor = module.LocalNEXGDDP_CountyProcessor

class XclimIndicatorsProcessor(LocalNEXGDDP_CountyProcessor):
    """
    Extended processor to calculate xclim indicators
    """
    
    def calculate_percentile_thresholds(self, ds, variable, percentiles=[10, 90]):
        """
        Calculate percentile thresholds from historical data
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Historical climate dataset
        variable : str
            Variable name (tas, tasmin, tasmax)
        percentiles : list
            Percentiles to calculate
        
        Returns:
        --------
        dict
            Percentile thresholds
        """
        print(f"Calculating {percentiles} percentiles for {variable}...")
        
        # Calculate percentiles along time dimension
        thresholds = {}
        for p in percentiles:
            threshold = ds[variable].quantile(p/100, dim='time')
            thresholds[f'p{p}'] = threshold
            
        return thresholds
    
    def process_xclim_indicators(self, scenario='ssp245', 
                               historical_period=(1980, 2010),
                               future_period=(2040, 2070)):
        """
        Calculate xclim indicators for all counties
        
        Parameters:
        -----------
        scenario : str
            Future scenario to process
        historical_period : tuple
            (start_year, end_year) for historical baseline
        future_period : tuple
            (start_year, end_year) for future projection
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all indicators by county and year
        """
        
        # First, load historical data to calculate percentile thresholds
        print("Loading historical data for threshold calculation...")
        hist_start, hist_end = historical_period
        
        # Load temperature data
        tas_hist = self.load_climate_data('tas', 'historical', hist_start, hist_end)
        tasmax_hist = self.load_climate_data('tasmax', 'historical', hist_start, hist_end)
        tasmin_hist = self.load_climate_data('tasmin', 'historical', hist_start, hist_end)
        pr_hist = self.load_climate_data('pr', 'historical', hist_start, hist_end)
        
        # Create county masks
        print("Creating county masks...")
        counties_mask = self.create_county_masks(tas_hist)
        mask_3D = counties_mask.mask_3D(tas_hist)
        
        # Process each county
        all_results = []
        
        for i, (idx, county) in enumerate(self.counties.iterrows()):
            geoid = county['GEOID']
            name = county['NAME']
            
            print(f"\nProcessing county {i+1}/{len(self.counties)}: {name}")
            
            # Extract county mask
            county_mask = mask_3D.isel(region=i)
            
            # Extract historical data for this county
            tas_county_hist = tas_hist['tas'].where(county_mask)
            tasmax_county_hist = tasmax_hist['tasmax'].where(county_mask)
            tasmin_county_hist = tasmin_hist['tasmin'].where(county_mask)
            pr_county_hist = pr_hist['pr'].where(county_mask)
            
            # Calculate spatial mean (weighted by latitude)
            weights = np.cos(np.deg2rad(tas_hist.lat))
            
            tas_mean_hist = tas_county_hist.weighted(weights).mean(dim=['lat', 'lon'])
            tasmax_mean_hist = tasmax_county_hist.weighted(weights).mean(dim=['lat', 'lon'])
            tasmin_mean_hist = tasmin_county_hist.weighted(weights).mean(dim=['lat', 'lon'])
            pr_mean_hist = pr_county_hist.weighted(weights).mean(dim=['lat', 'lon'])
            
            # Calculate percentile thresholds from historical data
            print("  Calculating percentile thresholds...")
            
            # For tx90p - 90th percentile of tasmax
            tasmax_p90 = tasmax_mean_hist.quantile(0.9, dim='time')
            
            # For tn10p - 10th percentile of tasmin
            tasmin_p10 = tasmin_mean_hist.quantile(0.1, dim='time')
            
            # For precipitation - day-of-year thresholds
            # Group by day of year and calculate threshold
            pr_doy = pr_mean_hist.groupby('time.dayofyear')
            pr_doy_thresh = pr_doy.quantile(0.95, dim='time')  # 95th percentile for each day
            
            # Now process each time period
            for period_name, (start_year, end_year), scen in [
                ('historical', historical_period, 'historical'),
                (scenario, future_period, scenario)
            ]:
                print(f"  Processing {period_name} ({start_year}-{end_year})...")
                
                # Load data for this period
                if scen == 'historical' and period_name == 'historical':
                    # Reuse already loaded data
                    tas_mean = tas_mean_hist
                    tasmax_mean = tasmax_mean_hist
                    tasmin_mean = tasmin_mean_hist
                    pr_mean = pr_mean_hist
                else:
                    # Load future scenario data
                    tas_data = self.load_climate_data('tas', scen, start_year, end_year)
                    tasmax_data = self.load_climate_data('tasmax', scen, start_year, end_year)
                    tasmin_data = self.load_climate_data('tasmin', scen, start_year, end_year)
                    pr_data = self.load_climate_data('pr', scen, start_year, end_year)
                    
                    # Extract county data
                    tas_county = tas_data['tas'].where(county_mask)
                    tasmax_county = tasmax_data['tasmax'].where(county_mask)
                    tasmin_county = tasmin_data['tasmin'].where(county_mask)
                    pr_county = pr_data['pr'].where(county_mask)
                    
                    # Calculate spatial means
                    tas_mean = tas_county.weighted(weights).mean(dim=['lat', 'lon'])
                    tasmax_mean = tasmax_county.weighted(weights).mean(dim=['lat', 'lon'])
                    tasmin_mean = tasmin_county.weighted(weights).mean(dim=['lat', 'lon'])
                    pr_mean = pr_county.weighted(weights).mean(dim=['lat', 'lon'])
                
                # Calculate xclim indicators
                print("  Calculating indicators...")
                
                # 1. tx90p - Percentage of days with tasmax > 90th percentile
                tx90p = atmos.tx90p(tasmax_mean, tasmax_p90, freq='YS')
                
                # 2. tx_days_above - Days with tasmax > 90°F (32.22°C)
                # Convert 90°F to Kelvin (305.37 K)
                tx_days_above = atmos.tx_days_above(tasmax_mean, thresh='305.37 K', freq='YS')
                
                # 3. tn10p - Percentage of days with tasmin < 10th percentile
                tn10p = atmos.tn10p(tasmin_mean, tasmin_p10, freq='YS')
                
                # 4. tn_days_below - Days with tasmin < 32°F (0°C)
                # Convert 32°F to Kelvin (273.15 K)
                tn_days_below = atmos.tn_days_below(tasmin_mean, thresh='273.15 K', freq='YS')
                
                # 5. tg_mean - Mean temperature
                tg_mean = atmos.tg_mean(tas_mean, freq='YS')
                
                # 6. days_over_precip_doy_thresh - Days with precip > 25.4mm
                # Convert 25.4mm/day to kg/m2/s (0.000294 kg/m2/s)
                days_over_precip = atmos.days_over_precip_thresh(
                    pr_mean, 
                    thresh='0.000294 kg m-2 s-1', 
                    freq='YS'
                )
                
                # 7. precip_accumulation - Total annual precipitation
                precip_accumulation = atmos.precip_accumulation(pr_mean, freq='YS')
                
                # Extract years
                years = pd.to_datetime(tx90p.time.values).year
                
                # Create results for each year
                for j, year in enumerate(years):
                    result = {
                        'GEOID': geoid,
                        'NAME': name,
                        'STATE': county.get('STATEFP', 'Unknown'),
                        'scenario': scen,
                        'year': year,
                        
                        # Temperature extremes
                        'tx90p_percent': float(tx90p.isel(time=j).values),
                        'tx_days_above_90F': float(tx_days_above.isel(time=j).values),
                        'tn10p_percent': float(tn10p.isel(time=j).values),
                        'tn_days_below_32F': float(tn_days_below.isel(time=j).values),
                        
                        # Mean temperature (convert from K to C)
                        'tg_mean_C': float(tg_mean.isel(time=j).values) - 273.15,
                        
                        # Precipitation
                        'days_precip_over_25.4mm': float(days_over_precip.isel(time=j).values),
                        'precip_accumulation_mm': float(precip_accumulation.isel(time=j).values) * 86400,  # Convert to mm
                    }
                    
                    all_results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        return df
    
    def save_xclim_results(self, df, output_prefix):
        """
        Save xclim indicator results
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Results dataframe
        output_prefix : str
            Output file prefix
        """
        # Save full results
        df.to_csv(f"{output_prefix}_xclim_indicators.csv", index=False)
        print(f"Results saved to {output_prefix}_xclim_indicators.csv")
        
        # Create summary by scenario
        scenario_summary = df.groupby(['scenario', 'year']).agg({
            'tx90p_percent': ['mean', 'std'],
            'tx_days_above_90F': ['mean', 'std'],
            'tn10p_percent': ['mean', 'std'],
            'tn_days_below_32F': ['mean', 'std'],
            'tg_mean_C': ['mean', 'std'],
            'days_precip_over_25.4mm': ['mean', 'std'],
            'precip_accumulation_mm': ['mean', 'std']
        }).reset_index()
        
        scenario_summary.to_csv(f"{output_prefix}_xclim_scenario_summary.csv", index=False)
        
        # Create state summary
        if 'STATE' in df.columns:
            state_summary = df.groupby(['STATE', 'scenario']).agg({
                'tx90p_percent': ['mean', 'std'],
                'tx_days_above_90F': ['mean', 'std'],
                'tn10p_percent': ['mean', 'std'],
                'tn_days_below_32F': ['mean', 'std'],
                'tg_mean_C': ['mean', 'std'],
                'days_precip_over_25.4mm': ['mean', 'std'],
                'precip_accumulation_mm': ['mean', 'std']
            }).reset_index()
            
            state_summary.to_csv(f"{output_prefix}_xclim_state_summary.csv", index=False)
        
        # Create change analysis
        hist_data = df[df['scenario'] == 'historical'].set_index(['GEOID', 'year'])
        future_data = df[df['scenario'] != 'historical'].set_index(['GEOID', 'year'])
        
        # Calculate average changes
        change_results = []
        for geoid in df['GEOID'].unique():
            county_name = df[df['GEOID'] == geoid]['NAME'].iloc[0]
            county_state = df[df['GEOID'] == geoid]['STATE'].iloc[0]
            
            hist_mean = hist_data.loc[geoid].mean()
            
            for scenario in df['scenario'].unique():
                if scenario != 'historical':
                    future_mean = future_data[future_data.index.get_level_values(0) == geoid].mean()
                    
                    change = {
                        'GEOID': geoid,
                        'NAME': county_name,
                        'STATE': county_state,
                        'scenario': scenario,
                        'tx90p_change': future_mean['tx90p_percent'] - hist_mean['tx90p_percent'],
                        'tx_days_above_90F_change': future_mean['tx_days_above_90F'] - hist_mean['tx_days_above_90F'],
                        'tn10p_change': future_mean['tn10p_percent'] - hist_mean['tn10p_percent'],
                        'tn_days_below_32F_change': future_mean['tn_days_below_32F'] - hist_mean['tn_days_below_32F'],
                        'tg_mean_C_change': future_mean['tg_mean_C'] - hist_mean['tg_mean_C'],
                        'days_precip_over_25.4mm_change': future_mean['days_precip_over_25.4mm'] - hist_mean['days_precip_over_25.4mm'],
                        'precip_accumulation_mm_change': future_mean['precip_accumulation_mm'] - hist_mean['precip_accumulation_mm']
                    }
                    change_results.append(change)
        
        change_df = pd.DataFrame(change_results)
        change_df.to_csv(f"{output_prefix}_xclim_changes.csv", index=False)
        
        print(f"Saved {len(df)} records across {len(df['GEOID'].unique())} counties")


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = XclimIndicatorsProcessor(
        counties_shapefile_path="/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    )
    
    # Calculate indicators
    df = processor.process_xclim_indicators(
        scenario='ssp245',
        historical_period=(1980, 2010),
        future_period=(2040, 2070)
    )
    
    # Save results
    processor.save_xclim_results(df, "county_climate_indicators")