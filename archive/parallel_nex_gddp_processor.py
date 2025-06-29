import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import tempfile
import shutil
import dask
import dask.array as da
from dask.distributed import Client, as_completed
import zarr
import h5py

warnings.filterwarnings('ignore')

class ParallelNEXGDDP_CountyProcessor:
    """
    Parallel processing of NEX-GDDP-CMIP6 climate data for county-level analysis
    """
    
    def __init__(self, counties_shapefile_path, base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"):
        """
        Initialize processor with county boundaries and data path
        """
        self.counties = gpd.read_file(counties_shapefile_path)
        self.counties = self.counties.to_crs('EPSG:4326')
        self.base_path = Path(base_data_path)
        
        # Pre-compute county masks for efficiency
        self.county_bounds = self._precompute_county_bounds()
        
        print(f"Initialized with {len(self.counties)} counties")
        print(f"Available cores: {mp.cpu_count()}")
    
    def _precompute_county_bounds(self):
        """Pre-compute bounding boxes for spatial subsetting"""
        bounds = {}
        for idx, county in self.counties.iterrows():
            geoid = county['GEOID']
            bounds[geoid] = county.geometry.bounds
        return bounds
    
    # Strategy 1: Convert to Zarr for parallel reads
    def convert_to_zarr(self, nc_file, zarr_file=None):
        """
        Convert NetCDF to Zarr format for parallel access
        
        Parameters:
        -----------
        nc_file : str
            Path to NetCDF file
        zarr_file : str, optional
            Output Zarr path (defaults to same name with .zarr)
        
        Returns:
        --------
        str
            Path to Zarr store
        """
        if zarr_file is None:
            zarr_file = str(nc_file).replace('.nc', '.zarr')
        
        if not Path(zarr_file).exists():
            print(f"Converting {nc_file} to Zarr format...")
            ds = xr.open_dataset(nc_file)
            ds.to_zarr(zarr_file, mode='w', consolidated=True)
            ds.close()
            print(f"Saved to {zarr_file}")
        
        return zarr_file
    
    # Strategy 2: Pre-load data into memory and share
    def load_data_to_memory(self, variable, scenario, start_year=None, end_year=None):
        """
        Load entire dataset into memory for shared access
        
        Parameters:
        -----------
        variable : str
            Climate variable
        scenario : str
            SSP scenario
        start_year : int
        end_year : int
        
        Returns:
        --------
        dict
            Dictionary with arrays and metadata
        """
        file_pattern = self.base_path / variable / scenario / f"{variable}_*.nc"
        files = list(file_pattern.parent.glob(file_pattern.name))
        
        print(f"Loading {variable} data into memory...")
        
        # Open and load to memory
        ds = xr.open_mfdataset(files, combine='by_coords') if len(files) > 1 else xr.open_dataset(files[0])
        
        if start_year and end_year:
            # Use isel to avoid calendar issues
            time_vals = ds.time.values
            if hasattr(time_vals[0], 'year'):
                # cftime objects
                mask = np.array([(t.year >= start_year and t.year <= end_year) for t in time_vals])
                ds = ds.isel(time=mask)
            else:
                # Standard datetime
                ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        
        # Convert to numpy arrays for efficient sharing
        data_dict = {
            'values': ds[variable].values,
            'lat': ds.lat.values,
            'lon': ds.lon.values,
            'time': ds.time.values,
            'units': ds[variable].attrs.get('units', 'unknown')
        }
        
        ds.close()
        return data_dict
    
    # Strategy 3: Process by chunks (spatial or temporal)
    def process_by_spatial_chunks(self, variable, scenario, n_chunks=4):
        """
        Process data in spatial chunks to avoid concurrent reads
        
        Parameters:
        -----------
        variable : str
            Climate variable
        scenario : str
            SSP scenario
        n_chunks : int
            Number of spatial chunks
        
        Returns:
        --------
        dict
            County results
        """
        # Divide counties into chunks
        county_chunks = np.array_split(self.counties.index, n_chunks)
        
        # Process each chunk in parallel
        with ProcessPoolExecutor(max_workers=n_chunks) as executor:
            futures = []
            for i, chunk_indices in enumerate(county_chunks):
                chunk_counties = self.counties.loc[chunk_indices]
                future = executor.submit(
                    self._process_county_chunk,
                    chunk_counties, variable, scenario, i
                )
                futures.append(future)
            
            # Collect results
            all_results = {}
            for future in futures:
                chunk_results = future.result()
                all_results.update(chunk_results)
        
        return all_results
    
    def _process_county_chunk(self, counties_chunk, variable, scenario, chunk_id):
        """Process a chunk of counties"""
        print(f"Processing chunk {chunk_id} with {len(counties_chunk)} counties")
        
        # Load data once for this chunk
        file_pattern = self.base_path / variable / scenario / f"{variable}_*.nc"
        files = list(file_pattern.parent.glob(file_pattern.name))
        
        ds = xr.open_mfdataset(files, combine='by_coords') if len(files) > 1 else xr.open_dataset(files[0])
        
        # Create masks for this chunk
        counties_mask = regionmask.from_geopandas(
            counties_chunk,
            names="GEOID",
            abbrevs="GEOID",
            name=f"Counties_Chunk_{chunk_id}"
        )
        
        mask_3D = counties_mask.mask_3D(ds)
        
        results = {}
        for i, (idx, county) in enumerate(counties_chunk.iterrows()):
            geoid = county['GEOID']
            county_mask = mask_3D.isel(region=i)
            county_data = ds[variable].where(county_mask)
            
            # Calculate weighted mean
            weights = np.cos(np.deg2rad(ds.lat))
            county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
            
            results[geoid] = {
                'name': county['NAME'],
                'state': county.get('STATE', 'Unknown'),
                'time': county_mean.time.values,
                'values': county_mean.values,
                'variable': variable,
                'units': ds[variable].attrs.get('units', 'unknown')
            }
        
        ds.close()
        return results
    
    # Strategy 4: Use Dask for lazy evaluation
    def process_with_dask(self, variable, scenario, start_year=None, end_year=None):
        """
        Process using Dask for parallel computation
        
        Parameters:
        -----------
        variable : str
            Climate variable
        scenario : str
            SSP scenario
        start_year : int
        end_year : int
        
        Returns:
        --------
        dict
            County results
        """
        # Initialize Dask client
        client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
        print(f"Dask dashboard: {client.dashboard_link}")
        
        try:
            # Open with Dask
            file_pattern = self.base_path / variable / scenario / f"{variable}_*.nc"
            files = list(file_pattern.parent.glob(file_pattern.name))
            
            # Open with chunks
            ds = xr.open_mfdataset(
                files, 
                combine='by_coords',
                chunks={'time': 365, 'lat': 100, 'lon': 100},
                parallel=True
            )
            
            if start_year and end_year:
                # Use isel to avoid calendar issues
                time_vals = ds.time.values
                if hasattr(time_vals[0], 'year'):
                    # cftime objects
                    mask = np.array([(t.year >= start_year and t.year <= end_year) for t in time_vals])
                    ds = ds.isel(time=mask)
                else:
                    # Standard datetime
                    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
            
            # Create county masks
            counties_mask = regionmask.from_geopandas(
                self.counties,
                names="GEOID",
                abbrevs="GEOID",
                name="US_Counties"
            )
            
            # Process counties in batches
            results = {}
            batch_size = 50
            
            for batch_start in range(0, len(self.counties), batch_size):
                batch_end = min(batch_start + batch_size, len(self.counties))
                batch_counties = self.counties.iloc[batch_start:batch_end]
                
                print(f"Processing counties {batch_start}-{batch_end}")
                
                # Create futures for this batch
                futures = []
                for idx, county in batch_counties.iterrows():
                    future = client.submit(
                        self._process_single_county_dask,
                        ds, county, variable, counties_mask
                    )
                    futures.append((county['GEOID'], future))
                
                # Collect results as they complete
                for geoid, future in futures:
                    result = future.result()
                    results[geoid] = result
            
            return results
            
        finally:
            client.close()
    
    def _process_single_county_dask(self, ds, county, variable, counties_mask):
        """Process single county with Dask"""
        # Get county geometry
        county_geom = county.geometry
        
        # Create mask for this county
        mask = counties_mask.mask(ds, lon_name='lon', lat_name='lat') == self.counties[self.counties['GEOID'] == county['GEOID']].index[0]
        
        # Apply mask and compute mean
        county_data = ds[variable].where(mask)
        weights = np.cos(np.deg2rad(ds.lat))
        county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
        
        # Compute the result
        result = {
            'name': county['NAME'],
            'state': county.get('STATE', 'Unknown'),
            'time': county_mean.time.values,
            'values': county_mean.compute().values,  # Trigger computation
            'variable': variable,
            'units': ds[variable].attrs.get('units', 'unknown')
        }
        
        return result
    
    # Strategy 5: Time-based parallelization
    def process_by_time_chunks(self, variable, scenario, years_per_chunk=5):
        """
        Process data in temporal chunks
        
        Parameters:
        -----------
        variable : str
            Climate variable
        scenario : str
            SSP scenario
        years_per_chunk : int
            Years per temporal chunk
        
        Returns:
        --------
        dict
            County results
        """
        # First, get the time range
        file_pattern = self.base_path / variable / scenario / f"{variable}_*.nc"
        files = list(file_pattern.parent.glob(file_pattern.name))
        
        # Quick open to get time info
        with xr.open_dataset(files[0]) as ds:
            # Handle cftime objects for non-standard calendars
            if hasattr(ds.time.values[0], 'year'):
                # cftime object
                start_year = ds.time.values[0].year
                end_year = ds.time.values[-1].year
            else:
                # Standard datetime
                start_year = pd.to_datetime(ds.time.values[0]).year
                end_year = pd.to_datetime(ds.time.values[-1]).year
        
        # Create time chunks
        time_chunks = []
        for year in range(start_year, end_year + 1, years_per_chunk):
            chunk_end = min(year + years_per_chunk - 1, end_year)
            time_chunks.append((year, chunk_end))
        
        print(f"Processing {len(time_chunks)} time chunks")
        
        # Process each time chunk in parallel
        with ProcessPoolExecutor(max_workers=min(len(time_chunks), mp.cpu_count())) as executor:
            futures = []
            for start, end in time_chunks:
                future = executor.submit(
                    self._process_time_chunk,
                    variable, scenario, start, end
                )
                futures.append((start, end, future))
            
            # Collect and merge results
            all_chunks = []
            for start, end, future in futures:
                chunk_results = future.result()
                all_chunks.append(chunk_results)
                print(f"Completed chunk {start}-{end}")
        
        # Merge temporal results
        return self._merge_temporal_results(all_chunks)
    
    def _process_time_chunk(self, variable, scenario, start_year, end_year):
        """Process a single time chunk"""
        # Load data for this time period
        data_dict = self.load_data_to_memory(variable, scenario, start_year, end_year)
        
        # Create masks
        counties_mask = regionmask.from_geopandas(
            self.counties,
            names="GEOID",
            abbrevs="GEOID",
            name="US_Counties"
        )
        
        # Create a temporary dataset from the data dict
        ds = xr.Dataset({
            variable: (['time', 'lat', 'lon'], data_dict['values'])
        }, coords={
            'time': data_dict['time'],
            'lat': data_dict['lat'],
            'lon': data_dict['lon']
        })
        
        mask_3D = counties_mask.mask_3D(ds)
        
        results = {}
        for i, (idx, county) in enumerate(self.counties.iterrows()):
            geoid = county['GEOID']
            county_mask = mask_3D.isel(region=i)
            county_data = ds[variable].where(county_mask)
            
            weights = np.cos(np.deg2rad(ds.lat))
            county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
            
            results[geoid] = {
                'name': county['NAME'],
                'state': county.get('STATE', 'Unknown'),
                'time': county_mean.time.values,
                'values': county_mean.values,
                'variable': variable,
                'units': data_dict['units']
            }
        
        return results
    
    def _merge_temporal_results(self, chunks):
        """Merge results from temporal chunks"""
        merged = {}
        
        # Get all counties from first chunk
        counties = list(chunks[0].keys())
        
        for geoid in counties:
            # Collect all time series for this county
            all_times = []
            all_values = []
            
            for chunk in chunks:
                if geoid in chunk:
                    all_times.extend(chunk[geoid]['time'])
                    all_values.extend(chunk[geoid]['values'])
            
            # Sort by time
            sorted_indices = np.argsort(all_times)
            
            merged[geoid] = {
                'name': chunks[0][geoid]['name'],
                'state': chunks[0][geoid]['state'],
                'time': np.array(all_times)[sorted_indices],
                'values': np.array(all_values)[sorted_indices],
                'variable': chunks[0][geoid]['variable'],
                'units': chunks[0][geoid]['units']
            }
        
        return merged
    
    # Main processing function with strategy selection
    def process_parallel(self, variable, scenario, strategy='spatial_chunks', **kwargs):
        """
        Process data using selected parallelization strategy
        
        Parameters:
        -----------
        variable : str
            Climate variable
        scenario : str
            SSP scenario
        strategy : str
            Parallelization strategy:
            - 'spatial_chunks': Process counties in spatial chunks
            - 'temporal_chunks': Process time periods in parallel
            - 'dask': Use Dask for lazy evaluation
            - 'memory': Load all data to memory first
            - 'zarr': Convert to Zarr format first
        **kwargs : dict
            Additional arguments for the selected strategy
        
        Returns:
        --------
        dict
            County results
        """
        print(f"Processing {variable} - {scenario} using {strategy} strategy")
        
        if strategy == 'spatial_chunks':
            return self.process_by_spatial_chunks(variable, scenario, 
                                                n_chunks=kwargs.get('n_chunks', 4))
        
        elif strategy == 'temporal_chunks':
            return self.process_by_time_chunks(variable, scenario,
                                             years_per_chunk=kwargs.get('years_per_chunk', 5))
        
        elif strategy == 'dask':
            return self.process_with_dask(variable, scenario,
                                        start_year=kwargs.get('start_year'),
                                        end_year=kwargs.get('end_year'))
        
        elif strategy == 'memory':
            # Load data to memory then process in parallel
            data_dict = self.load_data_to_memory(variable, scenario,
                                               kwargs.get('start_year'),
                                               kwargs.get('end_year'))
            # Process counties in parallel using the in-memory data
            return self._process_inmemory_parallel(data_dict, variable)
        
        elif strategy == 'zarr':
            # Convert to Zarr first
            nc_files = list((self.base_path / variable / scenario).glob("*.nc"))
            if nc_files:
                zarr_file = self.convert_to_zarr(nc_files[0])
                # Then process using Dask on Zarr
                return self._process_zarr_parallel(zarr_file, variable)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _process_inmemory_parallel(self, data_dict, variable):
        """Process counties in parallel using in-memory data"""
        # Create a shared memory array (for truly parallel processing)
        from multiprocessing import shared_memory
        
        # Create shared memory for the data
        shm = shared_memory.SharedMemory(create=True, size=data_dict['values'].nbytes)
        shared_array = np.ndarray(data_dict['values'].shape, dtype=data_dict['values'].dtype, buffer=shm.buf)
        shared_array[:] = data_dict['values'][:]
        
        try:
            # Process counties in parallel
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = []
                for idx, county in self.counties.iterrows():
                    future = executor.submit(
                        self._process_county_from_shared_memory,
                        county, variable, shm.name, 
                        data_dict['values'].shape, data_dict['values'].dtype,
                        data_dict['lat'], data_dict['lon'], data_dict['time'],
                        data_dict['units']
                    )
                    futures.append((county['GEOID'], future))
                
                # Collect results
                results = {}
                for geoid, future in futures:
                    results[geoid] = future.result()
                
                return results
                
        finally:
            shm.close()
            shm.unlink()
    
    def _process_county_from_shared_memory(self, county, variable, shm_name, shape, dtype, lat, lon, time, units):
        """Process single county from shared memory"""
        # Reconstruct shared array
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        # Create dataset
        ds = xr.Dataset({
            variable: (['time', 'lat', 'lon'], data)
        }, coords={
            'time': time,
            'lat': lat,
            'lon': lon
        })
        
        # Create mask for this county
        county_gdf = gpd.GeoDataFrame([county], crs='EPSG:4326')
        counties_mask = regionmask.from_geopandas(
            county_gdf,
            names="GEOID",
            abbrevs="GEOID",
            name="Single_County"
        )
        
        mask = counties_mask.mask(ds, lon_name='lon', lat_name='lat') == 0
        county_data = ds[variable].where(mask)
        
        # Calculate weighted mean
        weights = np.cos(np.deg2rad(ds.lat))
        county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
        
        result = {
            'name': county['NAME'],
            'state': county.get('STATE', 'Unknown'),
            'time': county_mean.time.values,
            'values': county_mean.values,
            'variable': variable,
            'units': units
        }
        
        existing_shm.close()
        return result


# Example usage with different strategies
if __name__ == "__main__":
    # Initialize processor
    processor = ParallelNEXGDDP_CountyProcessor(
        counties_shapefile_path="/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    )
    
    # Strategy 1: Spatial chunks (good for many counties)
    results = processor.process_parallel(
        variable='tas',
        scenario='ssp245',
        strategy='spatial_chunks',
        n_chunks=8  # Adjust based on your CPU cores
    )
    
    # Strategy 2: Temporal chunks (good for long time series)
    results = processor.process_parallel(
        variable='pr',
        scenario='historical',
        strategy='temporal_chunks',
        years_per_chunk=10
    )
    
    # Strategy 3: Dask (good for very large datasets)
    results = processor.process_parallel(
        variable='tasmax',
        scenario='ssp585',
        strategy='dask',
        start_year=2040,
        end_year=2070
    )
    
    # Strategy 4: In-memory (good for smaller datasets that fit in RAM)
    results = processor.process_parallel(
        variable='tasmin',
        scenario='ssp126',
        strategy='memory',
        start_year=2020,
        end_year=2030
    )
    
    # Strategy 5: Zarr conversion (good for repeated analysis)
    results = processor.process_parallel(
        variable='tas',
        scenario='ssp245',
        strategy='zarr'
    )
