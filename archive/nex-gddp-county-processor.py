import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class LocalNEXGDDP_CountyProcessor:
    """
    Process locally stored NEX-GDDP-CMIP6 climate data for county-level analysis
    """
    
    def __init__(self, counties_shapefile_path, base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"):
        """
        Initialize processor with county boundaries and data path
        
        Parameters:
        -----------
        counties_shapefile_path : str
            Path to county boundaries shapefile
        base_data_path : str
            Base path to climate data directory
        """
        self.counties = gpd.read_file(counties_shapefile_path)
        # Ensure CRS matches NEX-GDDP (WGS84)
        self.counties = self.counties.to_crs('EPSG:4326')
        self.base_path = Path(base_data_path)
        
        # Verify data directory exists
        if not self.base_path.exists():
            raise ValueError(f"Data directory {base_data_path} not found")
            
        print(f"Initialized with {len(self.counties)} counties")
        print(f"Data directory: {self.base_path}")
    
    def load_climate_data(self, variable, scenario, start_year=None, end_year=None):
        """
        Load climate data from local files
        
        Parameters:
        -----------
        variable : str
            Climate variable ('tas', 'pr', 'tasmax', 'tasmin')
        scenario : str
            SSP scenario ('historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585')
        start_year : int, optional
            Start year for time slice
        end_year : int, optional
            End year for time slice
        
        Returns:
        --------
        xarray.Dataset
            Climate data
        """
        # Construct file path - adjust pattern based on your file structure
        file_pattern = self.base_path / variable / scenario / f"{variable}_*.nc"
        files = list(file_pattern.parent.glob(file_pattern.name))
        
        if not files:
            raise FileNotFoundError(f"No files found for {variable} in {scenario}")
        
        print(f"Found {len(files)} files for {variable} - {scenario}")
        
        # Open dataset (handles multiple files if needed)
        if len(files) == 1:
            ds = xr.open_dataset(files[0])
        else:
            ds = xr.open_mfdataset(files, combine='by_coords')
        
        # Select time range if specified
        if start_year and end_year:
            ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        
        print(f"Loaded data shape: {ds[variable].shape}")
        print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        
        return ds
    
    def create_county_masks(self, ds):
        """
        Create regionmask for counties based on the dataset grid
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Climate dataset to get grid information
        
        Returns:
        --------
        regionmask.Regions
            County mask object
        """
        # Create regions from counties using GEOID as unique identifier
        counties_mask = regionmask.from_geopandas(
            self.counties,
            names="GEOID",  # Use GEOID as unique identifier
            abbrevs="GEOID",  # FIPS code column
            name="US_Counties"
        )
        
        return counties_mask
    
    def extract_county_statistics(self, ds, variable, counties_mask, return_annual=True):
        """
        Extract county-level statistics from gridded data
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Climate dataset
        variable : str
            Variable name
        counties_mask : regionmask.Regions
            County mask object
        return_annual : bool
            If True, return annual averages; if False, return full time series
        
        Returns:
        --------
        dict
            County statistics with time series
        """
        # Create 3D mask
        mask_3D = counties_mask.mask_3D(ds)
        
        # Initialize results
        county_results = {}
        
        # Process each county
        for i, (idx, county) in enumerate(self.counties.iterrows()):
            geoid = county['GEOID']
            name = county['NAME']
            
            print(f"Processing county {i+1}/{len(self.counties)}: {name}", end='\r')
            
            # Extract data for this county
            county_mask = mask_3D.isel(region=i)
            county_data = ds[variable].where(county_mask)
            
            # Calculate spatial mean (weighted by grid cell area if needed)
            if 'lat' in ds.dims and 'lon' in ds.dims:
                # Calculate weights based on latitude
                weights = np.cos(np.deg2rad(ds.lat))
                county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
            else:
                county_mean = county_data.mean(dim=['lat', 'lon'])
            
            # Convert to annual if requested
            if return_annual:
                # Resample to annual means
                annual_mean = county_mean.resample(time='YE').mean()
                time_values = annual_mean.time.values
                data_values = annual_mean.values
            else:
                time_values = county_mean.time.values
                data_values = county_mean.values
            
            # Store results
            county_results[geoid] = {
                'name': name,
                'state': county.get('STATEFP', 'Unknown'),
                'statefp': county.get('STATEFP', 'Unknown'),
                'time': time_values,
                'values': data_values,
                'variable': variable,
                'units': ds[variable].attrs.get('units', 'unknown')
            }
        
        print("\nCounty extraction complete!")
        return county_results
    
    def process_scenario_comparison(self, variable, scenarios=['historical', 'ssp245', 'ssp585'], 
                                  historical_period=(1980, 2010), future_period=(2040, 2070)):
        """
        Process and compare multiple scenarios
        
        Parameters:
        -----------
        variable : str
            Climate variable
        scenarios : list
            List of scenarios to process
        historical_period : tuple
            (start_year, end_year) for historical baseline
        future_period : tuple
            (start_year, end_year) for future projection
        
        Returns:
        --------
        pandas.DataFrame
            Comparison results by county
        """
        results = {}
        
        for scenario in scenarios:
            print(f"\nProcessing {scenario} scenario...")
            
            # Determine time period
            if scenario == 'historical':
                start_year, end_year = historical_period
            else:
                start_year, end_year = future_period
            
            # Load data
            try:
                ds = self.load_climate_data(variable, scenario, start_year, end_year)
                
                # Create county masks
                counties_mask = self.create_county_masks(ds)
                
                # Extract county statistics with annual data
                county_stats = self.extract_county_statistics(ds, variable, counties_mask, return_annual=True)
                
                # Store results
                results[scenario] = county_stats
                
            except Exception as e:
                print(f"Error processing {scenario}: {e}")
                continue
        
        # Create both comparison and annual dataframes
        comparison_df = self.create_comparison_dataframe(results, variable)
        annual_df = self.create_annual_dataframe(results, variable)
        
        return comparison_df, annual_df
    
    def create_comparison_dataframe(self, results, variable):
        """
        Create a comparison dataframe from scenario results
        
        Parameters:
        -----------
        results : dict
            Results by scenario
        variable : str
            Climate variable
        
        Returns:
        --------
        pandas.DataFrame
            Comparison dataframe
        """
        rows = []
        
        # Get all counties from the first available scenario
        scenario_keys = list(results.keys())
        if not scenario_keys:
            return pd.DataFrame()
        
        counties = results[scenario_keys[0]].keys()
        
        for geoid in counties:
            row = {'GEOID': geoid}
            
            # Add basic info from first scenario
            if scenario_keys:
                row['NAME'] = results[scenario_keys[0]][geoid]['name']
                row['STATE'] = results[scenario_keys[0]][geoid]['state']
            
            # Process each scenario
            for scenario in results:
                if geoid in results[scenario]:
                    data = results[scenario][geoid]
                    values = data['values']
                    
                    # Calculate statistics
                    if variable == 'pr':
                        # Convert precipitation from kg/m2/s to mm/year
                        annual_values = values * 86400 * 365.25
                        row[f'{scenario}_mean_annual_mm'] = np.mean(annual_values)
                        row[f'{scenario}_cv'] = np.std(annual_values) / np.mean(annual_values)
                    else:
                        # Temperature variables - convert from K to C
                        celsius_values = values - 273.15
                        row[f'{scenario}_mean_C'] = np.mean(celsius_values)
                        row[f'{scenario}_trend_C_per_decade'] = self.calculate_trend(
                            data['time'], celsius_values
                        )
                        
                        if variable == 'tasmax':
                            row[f'{scenario}_days_above_35C'] = np.sum(celsius_values > 35) / len(celsius_values) * 365.25
                        elif variable == 'tasmin':
                            row[f'{scenario}_frost_days'] = np.sum(celsius_values < 0) / len(celsius_values) * 365.25
            
            # Calculate changes if we have historical and future scenarios
            if 'historical' in results and 'ssp245' in results:
                if variable == 'pr':
                    hist_val = row.get('historical_mean_annual_mm', np.nan)
                    fut_val = row.get('ssp245_mean_annual_mm', np.nan)
                    if not np.isnan(hist_val) and not np.isnan(fut_val):
                        row['change_ssp245_mm'] = fut_val - hist_val
                        row['change_ssp245_percent'] = (fut_val - hist_val) / hist_val * 100
                else:
                    hist_val = row.get('historical_mean_C', np.nan)
                    fut_val = row.get('ssp245_mean_C', np.nan)
                    if not np.isnan(hist_val) and not np.isnan(fut_val):
                        row['change_ssp245_C'] = fut_val - hist_val
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_annual_dataframe(self, results, variable):
        """
        Create an annual dataframe with one row per county per year
        
        Parameters:
        -----------
        results : dict
            Results by scenario
        variable : str
            Climate variable
        
        Returns:
        --------
        pandas.DataFrame
            Annual dataframe
        """
        rows = []
        
        for scenario, scenario_data in results.items():
            for geoid, county_data in scenario_data.items():
                # Get years from time values
                years = pd.to_datetime(county_data['time']).year
                
                # Create one row per year
                for i, year in enumerate(years):
                    row = {
                        'GEOID': geoid,
                        'NAME': county_data['name'],
                        'STATE': county_data['state'],
                        'STATEFP': county_data['statefp'],
                        'scenario': scenario,
                        'year': year,
                        'variable': variable,
                        'units': county_data['units']
                    }
                    
                    # Add value based on variable type
                    if variable == 'pr':
                        # Convert precipitation from kg/m2/s to mm/year
                        row['value_mm_year'] = county_data['values'][i] * 86400 * 365.25
                    else:
                        # Temperature variables - convert from K to C
                        row['value_celsius'] = county_data['values'][i] - 273.15
                    
                    # Add raw value
                    row['value_raw'] = county_data['values'][i]
                    
                    rows.append(row)
        
        # Create dataframe and sort
        df = pd.DataFrame(rows)
        df = df.sort_values(['GEOID', 'scenario', 'year'])
        
        return df
    
    def calculate_trend(self, time, values):
        """
        Calculate linear trend per decade
        
        Parameters:
        -----------
        time : array
            Time values
        values : array
            Data values
        
        Returns:
        --------
        float
            Trend per decade
        """
        # Convert time to years
        years = pd.to_datetime(time).year + pd.to_datetime(time).dayofyear / 365.25
        
        # Calculate trend
        if len(years) > 1:
            trend = np.polyfit(years, values, 1)[0]
            return trend * 10  # Per decade
        else:
            return np.nan
    
    def save_results(self, df, output_prefix, annual_df=None):
        """
        Save results to CSV and create summary statistics
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Summary results dataframe
        output_prefix : str
            Output file prefix
        annual_df : pandas.DataFrame, optional
            Annual results dataframe
        """
        # Save summary results
        df.to_csv(f"{output_prefix}_summary.csv", index=False)
        print(f"Summary results saved to {output_prefix}_summary.csv")
        
        # Save annual results if provided
        if annual_df is not None and not annual_df.empty:
            annual_df.to_csv(f"{output_prefix}_annual.csv", index=False)
            print(f"Annual results saved to {output_prefix}_annual.csv")
            
            # Create state-year summary
            if 'STATE' in annual_df.columns and 'year' in annual_df.columns:
                state_year_summary = annual_df.groupby(['STATE', 'year', 'scenario']).agg({
                    'value_celsius' if 'value_celsius' in annual_df.columns else 'value_mm_year': ['mean', 'std', 'min', 'max']
                }).reset_index()
                state_year_summary.to_csv(f"{output_prefix}_state_year_summary.csv", index=False)
                print(f"State-year summary saved to {output_prefix}_state_year_summary.csv")
        
        # Create state summary from main dataframe
        if 'STATE' in df.columns:
            state_summary = df.groupby('STATE').agg({
                col: ['mean', 'std', 'min', 'max'] 
                for col in df.columns if col not in ['GEOID', 'NAME', 'STATE']
            })
            state_summary.to_csv(f"{output_prefix}_state_summary.csv")
            print(f"State summary saved to {output_prefix}_state_summary.csv")
    
    def create_climate_report(self, output_prefix='climate_analysis'):
        """
        Create a comprehensive climate report for all variables and scenarios
        
        Parameters:
        -----------
        output_prefix : str
            Prefix for output files
        """
        variables = ['tas', 'tasmin', 'tasmax', 'pr']
        scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
        
        all_results = {}
        
        for var in variables:
            print(f"\n{'='*50}")
            print(f"Processing variable: {var}")
            print(f"{'='*50}")
            
            try:
                df, annual_df = self.process_scenario_comparison(
                    variable=var,
                    scenarios=scenarios,
                    historical_period=(1980, 2010),
                    future_period=(2040, 2070)
                )
                
                # Save variable-specific results
                self.save_results(df, f"{output_prefix}_{var}", annual_df)
                all_results[var] = (df, annual_df)
                
            except Exception as e:
                print(f"Error processing {var}: {e}")
                continue
        
        return all_results


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = LocalNEXGDDP_CountyProcessor(
        counties_shapefile_path="/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    )
    
    # Process single variable for specific scenarios
    summary_df, annual_df = processor.process_scenario_comparison(
        variable='tas',
        scenarios=['historical', 'ssp245', 'ssp585'],
        historical_period=(1980, 2010),
        future_period=(2040, 2070)
    )
    
    # Save results
    processor.save_results(summary_df, "county_temperature_analysis", annual_df)
    
    # Or create comprehensive report for all variables
    # all_results = processor.create_climate_report("noresm2_county_climate")
