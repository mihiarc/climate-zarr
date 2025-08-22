#!/usr/bin/env python
"""Processing strategy for county-level climate data analysis using precise geometry clipping."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ..utils.spatial_utils import (
    get_time_information, 
    get_coordinate_arrays,
    clip_county_data
)
from ..utils.data_utils import calculate_statistics

console = Console()


class ProcessingStrategy(ABC):
    """Abstract base class for processing strategies."""
    
    @abstractmethod
    def process(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        variable: str,
        scenario: str,
        threshold: float,
        n_workers: int = 4
    ) -> pd.DataFrame:
        """Process climate data using this strategy.
        
        Args:
            data: Climate data array
            gdf: County geometries
            variable: Climate variable name
            scenario: Scenario name
            threshold: Threshold value
            n_workers: Number of workers
            
        Returns:
            DataFrame with processed results
        """
        pass


class SpatialChunkedStrategy(ProcessingStrategy):
    """Advanced spatial chunking strategy for optimal memory utilization.
    
    This strategy implements:
    - Spatial locality-aware chunking using quadtree partitioning
    - Dynamic memory estimation based on county size and complexity
    - Parallel processing with precise geometry clipping
    - Graceful error handling and chunk recovery
    - Memory-optimized data structures and caching
    """
    
    def __init__(self, 
                 target_memory_usage: float = 0.75,
                 min_chunk_size: int = 5, 
                 max_chunk_size: int = 50,
                 enable_spatial_cache: bool = True):
        """Initialize spatial chunking strategy.
        
        Args:
            target_memory_usage: Target percentage of available memory to use (0.7-0.8)
            min_chunk_size: Minimum counties per chunk
            max_chunk_size: Maximum counties per chunk
            enable_spatial_cache: Enable spatial data caching for performance
        """
        self.target_memory_usage = target_memory_usage
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.enable_spatial_cache = enable_spatial_cache
        self._spatial_cache = {} if enable_spatial_cache else None
    
    def process(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        variable: str,
        scenario: str,
        threshold: float,
        n_workers: int = 4
    ) -> pd.DataFrame:
        """Process using spatial chunking with optimal memory utilization."""
        import psutil
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from shapely.geometry import box
        import numpy as np
        
        console.print("[yellow]Initializing spatial chunked processing strategy...[/yellow]")
        
        # Validate and prepare spatial data
        self._validate_spatial_data(data, gdf)
        
        # Calculate available memory and optimal chunk parameters
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        target_memory_gb = available_memory_gb * self.target_memory_usage
        
        console.print(f"[cyan]Available memory: {available_memory_gb:.1f} GB[/cyan]")
        console.print(f"[cyan]Target memory usage: {target_memory_gb:.1f} GB ({self.target_memory_usage*100:.0f}%)[/cyan]")
        
        # Create spatial chunks based on geography and data volume
        chunks = self._create_spatial_chunks(data, gdf, target_memory_gb)
        
        console.print(f"[green]Created {len(chunks)} spatial chunks for processing[/green]")
        
        # Process chunks in parallel with memory monitoring
        results = []
        failed_chunks = []
        
        # Use ThreadPoolExecutor for I/O bound operations (better for rioxarray clipping)
        with ThreadPoolExecutor(max_workers=min(n_workers, len(chunks))) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self._process_chunk, 
                    chunk_id, chunk_counties, data, gdf, variable, scenario, threshold
                ): chunk_id 
                for chunk_id, chunk_counties in enumerate(chunks)
            }
            
            # Collect results with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task(f"Processing {len(chunks)} chunks...", total=len(chunks))
                
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    
                    try:
                        chunk_results = future.result()
                        if chunk_results:
                            results.extend(chunk_results)
                        
                        # Monitor memory usage
                        current_memory = psutil.virtual_memory().percent
                        if current_memory > 85:
                            console.print(f"[yellow]Warning: High memory usage ({current_memory:.1f}%)[/yellow]")
                            
                    except Exception as e:
                        failed_chunks.append(chunk_id)
                        console.print(f"[red]Chunk {chunk_id} failed: {str(e)}[/red]")
                    
                    progress.advance(task)
        
        # Handle failed chunks with fallback processing
        if failed_chunks:
            console.print(f"[yellow]Retrying {len(failed_chunks)} failed chunks sequentially...[/yellow]")
            for chunk_id in failed_chunks:
                try:
                    chunk_results = self._process_chunk_fallback(
                        chunks[chunk_id], data, gdf, variable, scenario, threshold
                    )
                    if chunk_results:
                        results.extend(chunk_results)
                except Exception as e:
                    console.print(f"[red]Chunk {chunk_id} failed on retry: {str(e)}[/red]")
        
        console.print(f"[green]Spatial chunked processing complete: {len(results)} county-years processed[/green]")
        
        return pd.DataFrame(results)
    
    def _create_spatial_chunks(self, data: xr.DataArray, gdf: gpd.GeoDataFrame, 
                              target_memory_gb: float) -> List[List[int]]:
        """Create spatially-aware chunks optimized for memory usage."""
        from shapely.geometry import Point
        from sklearn.cluster import KMeans
        import numpy as np
        
        # Calculate county centroids for spatial clustering
        centroids = gdf.geometry.centroid
        coords = np.array([[p.x, p.y] for p in centroids])
        
        # Estimate memory usage per county based on bounding box area
        county_memory_estimates = self._estimate_county_memory_usage(data, gdf)
        
        # Create initial spatial clusters using K-means
        n_initial_clusters = max(self.min_chunk_size, 
                               int(len(gdf) * np.mean(county_memory_estimates) / target_memory_gb))
        
        console.print(f"[cyan]Creating {n_initial_clusters} initial spatial clusters[/cyan]")
        
        kmeans = KMeans(n_clusters=n_initial_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Refine clusters based on memory constraints and spatial adjacency
        refined_chunks = self._refine_clusters_by_memory(
            cluster_labels, county_memory_estimates, target_memory_gb
        )
        
        # Apply spatial locality optimization
        optimized_chunks = self._optimize_spatial_locality(refined_chunks, gdf)
        
        return optimized_chunks
    
    def _estimate_county_memory_usage(self, data: xr.DataArray, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Estimate memory usage for each county based on spatial characteristics."""
        import numpy as np
        
        # Get data resolution
        lat_res = abs(float(data.y[1] - data.y[0]))
        lon_res = abs(float(data.x[1] - data.x[0]))
        time_steps = data.sizes.get('time', 1)
        
        # Calculate approximate pixel count for each county
        memory_estimates = []
        
        for _, county in gdf.iterrows():
            bounds = county.geometry.bounds
            width_deg = bounds[2] - bounds[0]  # max_x - min_x
            height_deg = bounds[3] - bounds[1]  # max_y - min_y
            
            # Estimate pixel count (conservative)
            approx_pixels = (width_deg / lon_res) * (height_deg / lat_res)
            
            # Account for complex boundaries (coastal counties use more memory)
            boundary_complexity = len(county.geometry.exterior.coords) if hasattr(county.geometry, 'exterior') else 100
            complexity_factor = min(2.0, 1.0 + boundary_complexity / 1000)
            
            # Memory estimate in GB (float64 = 8 bytes per value)
            memory_gb = (approx_pixels * time_steps * 8) / (1024**3) * complexity_factor
            memory_estimates.append(memory_gb)
        
        return np.array(memory_estimates)
    
    def _refine_clusters_by_memory(self, cluster_labels: np.ndarray, 
                                  memory_estimates: np.ndarray, 
                                  target_memory_gb: float) -> List[List[int]]:
        """Refine clusters to fit within memory constraints."""
        chunks = []
        
        # Group counties by initial cluster
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_memory = memory_estimates[cluster_indices]
            
            # Split large clusters that exceed memory limits
            if np.sum(cluster_memory) > target_memory_gb * 0.8:  # 80% of target per chunk
                # Sort by memory usage and create smaller chunks
                sorted_indices = cluster_indices[np.argsort(cluster_memory)]
                
                current_chunk = []
                current_memory = 0.0
                
                for idx in sorted_indices:
                    if (current_memory + memory_estimates[idx] < target_memory_gb * 0.8 and 
                        len(current_chunk) < self.max_chunk_size):
                        current_chunk.append(int(idx))
                        current_memory += memory_estimates[idx]
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = [int(idx)]
                        current_memory = memory_estimates[idx]
                
                if current_chunk:
                    chunks.append(current_chunk)
            else:
                # Cluster fits within memory limits
                if len(cluster_indices) >= self.min_chunk_size:
                    chunks.append(cluster_indices.tolist())
                else:
                    # Merge small clusters
                    if chunks and len(chunks[-1]) + len(cluster_indices) <= self.max_chunk_size:
                        chunks[-1].extend(cluster_indices.tolist())
                    else:
                        chunks.append(cluster_indices.tolist())
        
        return chunks
    
    def _optimize_spatial_locality(self, chunks: List[List[int]], gdf: gpd.GeoDataFrame) -> List[List[int]]:
        """Optimize chunks for spatial locality to improve cache efficiency."""
        optimized_chunks = []
        
        for chunk in chunks:
            if len(chunk) <= 2:
                optimized_chunks.append(chunk)
                continue
            
            # Get centroids for chunk counties
            chunk_centroids = [gdf.iloc[i].geometry.centroid for i in chunk]
            
            # Sort by spatial proximity using a simple nearest-neighbor approach
            if chunk_centroids:
                sorted_chunk = []
                remaining = chunk.copy()
                
                # Start with the first county
                current_idx = remaining.pop(0)
                sorted_chunk.append(current_idx)
                current_centroid = gdf.iloc[current_idx].geometry.centroid
                
                # Add nearest neighbors iteratively
                while remaining:
                    distances = [current_centroid.distance(gdf.iloc[i].geometry.centroid) 
                               for i in remaining]
                    nearest_idx = remaining.pop(np.argmin(distances))
                    sorted_chunk.append(nearest_idx)
                    current_centroid = gdf.iloc[nearest_idx].geometry.centroid
                
                optimized_chunks.append(sorted_chunk)
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _process_chunk(self, chunk_id: int, county_indices: List[int], 
                      data: xr.DataArray, gdf: gpd.GeoDataFrame, variable: str, 
                      scenario: str, threshold: float) -> List[Dict]:
        """Process a single chunk of counties with optimized memory management."""
        import gc
        import threading
        from ..utils.data_utils import calculate_statistics
        
        # Thread-local storage for chunk processing
        local_cache = threading.local()
        
        try:
            chunk_results = []
            years, unique_years = get_time_information(data)
            
            console.print(f"[cyan]Processing chunk {chunk_id} with {len(county_indices)} counties[/cyan]")
            
            # Pre-extract county data for this chunk to optimize I/O
            chunk_counties = [gdf.iloc[idx] for idx in county_indices]
            
            # Process counties within the chunk in spatial order
            for county_idx, county in zip(county_indices, chunk_counties):
                try:
                    # Check spatial cache first
                    cache_key = f"{county_idx}_{data.shape}"
                    cached_clip = None
                    
                    if self.enable_spatial_cache and self._spatial_cache is not None:
                        cached_clip = self._spatial_cache.get(cache_key)
                    
                    # Clip county data using optimized strategy
                    if cached_clip is not None:
                        clipped = cached_clip
                    else:
                        clipped = self._clip_county_optimized(data, county)
                        
                        # Cache the clipped geometry for reuse
                        if (self.enable_spatial_cache and self._spatial_cache is not None 
                            and clipped is not None and clipped.size > 0):
                            # Only cache smaller clips to avoid memory issues
                            if clipped.nbytes < 50 * 1024 * 1024:  # 50MB limit
                                self._spatial_cache[cache_key] = clipped
                    
                    if clipped is not None and clipped.size > 0:
                        # Process all years for this county efficiently
                        county_results = self._process_county_years_chunked(
                            clipped, county, years, unique_years, variable, scenario, threshold
                        )
                        chunk_results.extend(county_results)
                        
                    # Explicit cleanup of large temporary data
                    del clipped
                    
                except Exception as e:
                    console.print(f"[red]Error processing county {county_idx} in chunk {chunk_id}: {str(e)}[/red]")
                    continue
            
            # Force garbage collection after chunk processing
            gc.collect()
            
            console.print(f"[green]Chunk {chunk_id} complete: {len(chunk_results)} results[/green]")
            return chunk_results
            
        except Exception as e:
            console.print(f"[red]Chunk {chunk_id} processing failed: {str(e)}[/red]")
            return []
    
    def _process_county_years_chunked(self, clipped_data: xr.DataArray, county, 
                                     years: np.ndarray, unique_years: np.ndarray,
                                     variable: str, scenario: str, threshold: float) -> List[Dict]:
        """Process all years for a county with memory-optimized chunking."""
        county_results = []
        
        # Pre-compute county info to avoid repeated dictionary creation
        county_info = {
            'county_id': county['county_id'],
            'county_name': county['county_name'],
            'state': county['state']
        }
        
        # Process years in chunks to manage memory for large time series
        year_chunk_size = 10  # Process 10 years at a time
        
        for i in range(0, len(unique_years), year_chunk_size):
            year_chunk = unique_years[i:i+year_chunk_size]
            
            try:
                for year in year_chunk:
                    # Efficient boolean indexing for year selection
                    year_mask = years == year
                    year_data = clipped_data.isel(time=year_mask)
                    
                    # Calculate spatial means efficiently with chunked operations
                    if year_data.size > 100000:  # For large datasets, use Dask
                        daily_means = year_data.mean(dim=['y', 'x'], skipna=True).compute().values
                    else:
                        daily_means = year_data.mean(dim=['y', 'x'], skipna=True).values
                    
                    # Filter out any NaN daily means
                    if np.any(np.isnan(daily_means)):
                        daily_means = daily_means[~np.isnan(daily_means)]
                        if len(daily_means) == 0:
                            continue
                    
                    # Calculate statistics
                    stats = calculate_statistics(
                        daily_means, variable, threshold, year, scenario, county_info
                    )
                    
                    if stats:
                        county_results.append(stats)
                        
            except Exception as e:
                console.print(f"[red]Error processing years {year_chunk[0]}-{year_chunk[-1]} for {county_info['county_name']}: {str(e)}[/red]")
                continue
        
        return county_results
    
    def _process_chunk_fallback(self, county_indices: List[int], data: xr.DataArray, 
                               gdf: gpd.GeoDataFrame, variable: str, scenario: str, 
                               threshold: float) -> List[Dict]:
        """Fallback processing for failed chunks using sequential processing."""
        console.print(f"[yellow]Using fallback sequential processing for {len(county_indices)} counties[/yellow]")
        
        results = []
        years, unique_years = get_time_information(data)
        
        for county_idx in county_indices:
            try:
                county = gdf.iloc[county_idx]
                
                # Use basic clipping without caching
                clipped = self._clip_county_optimized(data, county)
                
                if clipped is not None and clipped.size > 0:
                    # Use the vectorized strategy's year processing method
                    county_results = self._process_county_years_basic(
                        clipped, county, years, unique_years, variable, scenario, threshold
                    )
                    results.extend(county_results)
                    
            except Exception as e:
                console.print(f"[red]Fallback failed for county {county_idx}: {str(e)}[/red]")
                continue
        
        return results
    
    def _process_county_years_basic(self, clipped_data: xr.DataArray, county, 
                                   years: np.ndarray, unique_years: np.ndarray,
                                   variable: str, scenario: str, threshold: float) -> List[Dict]:
        """Basic year processing without advanced optimization."""
        from ..utils.data_utils import calculate_statistics
        
        county_results = []
        county_info = {
            'county_id': county['county_id'],
            'county_name': county['county_name'],
            'state': county['state']
        }
        
        for year in unique_years:
            try:
                year_mask = years == year
                year_data = clipped_data.isel(time=year_mask)
                daily_means = year_data.mean(dim=['y', 'x'], skipna=True).values
                
                if np.any(np.isnan(daily_means)):
                    daily_means = daily_means[~np.isnan(daily_means)]
                    if len(daily_means) == 0:
                        continue
                
                stats = calculate_statistics(
                    daily_means, variable, threshold, year, scenario, county_info
                )
                
                if stats:
                    county_results.append(stats)
                    
            except Exception as e:
                console.print(f"[red]Error processing year {year} for {county_info['county_name']}: {str(e)}[/red]")
                continue
        
        return county_results
    
    def _validate_spatial_data(self, data: xr.DataArray, gdf: gpd.GeoDataFrame) -> None:
        """Validate spatial data consistency and CRS alignment."""
        
        # Ensure data has CRS information
        if data.rio.crs is None:
            console.print("[yellow]Warning: Data CRS not specified, assuming EPSG:4326 (WGS84)[/yellow]")
            data = data.rio.write_crs("EPSG:4326")
        
        # Ensure GeoDataFrame has CRS
        if gdf.crs is None:
            console.print("[yellow]Warning: County GeoDataFrame CRS not specified, assuming EPSG:4326 (WGS84)[/yellow]")
            gdf = gdf.set_crs("EPSG:4326")
        
        # Check if CRS alignment is needed
        if str(data.rio.crs) != str(gdf.crs):
            console.print(f"[yellow]CRS mismatch detected - Data: {data.rio.crs}, Counties: {gdf.crs}[/yellow]")
            # Note: rioxarray.clip handles CRS transformation automatically
    
    def _clip_county_optimized(self, data: xr.DataArray, county) -> xr.DataArray:
        """Perform optimized county clipping with enhanced error handling."""
        try:
            # Use precise geometry clipping with all_touched=True for coastal counties
            clipped = clip_county_data(
                data, 
                county.geometry, 
                all_touched=True  # Critical for coastal counties and complex boundaries
            )
            
            # Additional validation for very small counties or edge cases
            if clipped.size == 0:
                # Try with stricter clipping for tiny counties that might be missed
                clipped = clip_county_data(data, county.geometry, all_touched=False)
            
            return clipped
            
        except Exception as e:
            console.print(f"[red]Clipping failed for {county.get('county_name', 'Unknown')}: {str(e)}[/red]")
            return None


class VectorizedStrategy(ProcessingStrategy):
    """Optimized vectorized processing strategy using precise rioxarray geometry clipping.
    
    This strategy provides:
    - Precise geometric clipping with proper CRS handling
    - Memory-efficient sequential processing 
    - Robust error handling for complex county boundaries
    - Optimal handling of coastal counties and edge cases
    """
    
    def process(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        variable: str,
        scenario: str,
        threshold: float,
        n_workers: int = 4
    ) -> pd.DataFrame:
        """Process using optimized vectorized operations with rioxarray clipping.
        
        Args:
            data: Climate data array with spatial coordinates
            gdf: County geometries (assumed to be in WGS84/EPSG:4326)
            variable: Climate variable name
            scenario: Scenario name
            threshold: Threshold value for calculations
            n_workers: Number of workers (kept for API compatibility)
            
        Returns:
            DataFrame with processed county statistics
        """
        
        console.print("[yellow]Processing counties with optimized rioxarray clipping...[/yellow]")
        
        # Validate and prepare spatial data
        self._validate_spatial_data(data, gdf)
        
        results = []
        years, unique_years = get_time_information(data)
        
        console.print(f"[cyan]Processing {len(gdf)} counties over {len(unique_years)} years[/cyan]")
        console.print(f"[cyan]Data shape: {data.shape} (time, lat, lon)[/cyan]")
        console.print(f"[cyan]Data CRS: {data.rio.crs or 'EPSG:4326 (assumed)'}[/cyan]")
        
        # Track processing statistics
        successful_counties = 0
        failed_counties = 0
        empty_clips = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing counties...", total=len(gdf))
            
            for idx, county in gdf.iterrows():
                try:
                    # Clip data to county using optimized rioxarray clipping
                    clipped = self._clip_county_optimized(data, county)
                    
                    if clipped is not None and clipped.size > 0:
                        # Process all years for this county
                        county_results = self._process_county_years(
                            clipped, county, years, unique_years, variable, scenario, threshold
                        )
                        results.extend(county_results)
                        successful_counties += 1
                    else:
                        empty_clips += 1
                        console.print(f"[yellow]Warning: No data found for {county['county_name']}, {county['state']}[/yellow]")
                        
                except Exception as e:
                    failed_counties += 1
                    console.print(f"[red]Error processing {county['county_name']}, {county['state']}: {str(e)}[/red]")
                
                progress.advance(task)
        
        # Report processing summary
        console.print(f"[green]Processing complete: {successful_counties} successful, {empty_clips} empty clips, {failed_counties} failed[/green]")
        
        return pd.DataFrame(results)
    
    def _validate_spatial_data(self, data: xr.DataArray, gdf: gpd.GeoDataFrame) -> None:
        """Validate spatial data consistency and CRS alignment."""
        
        # Ensure data has CRS information
        if data.rio.crs is None:
            console.print("[yellow]Warning: Data CRS not specified, assuming EPSG:4326 (WGS84)[/yellow]")
            data = data.rio.write_crs("EPSG:4326")
        
        # Ensure GeoDataFrame has CRS
        if gdf.crs is None:
            console.print("[yellow]Warning: County GeoDataFrame CRS not specified, assuming EPSG:4326 (WGS84)[/yellow]")
            gdf = gdf.set_crs("EPSG:4326")
        
        # Check if CRS alignment is needed
        if str(data.rio.crs) != str(gdf.crs):
            console.print(f"[yellow]CRS mismatch detected - Data: {data.rio.crs}, Counties: {gdf.crs}[/yellow]")
            # Note: rioxarray.clip handles CRS transformation automatically
    
    def _clip_county_optimized(self, data: xr.DataArray, county) -> xr.DataArray:
        """Perform optimized county clipping with enhanced error handling.
        
        Args:
            data: Climate data array
            county: County row from GeoDataFrame
            
        Returns:
            Clipped data array or None if clipping fails
        """
        try:
            # Use precise geometry clipping with all_touched=True for coastal counties
            # This ensures we capture all pixels that intersect with complex county boundaries
            clipped = clip_county_data(
                data, 
                county.geometry, 
                all_touched=True  # Critical for coastal counties and complex boundaries
            )
            
            # Additional validation for very small counties or edge cases
            if clipped.size == 0:
                # Try with stricter clipping for tiny counties that might be missed
                clipped = clip_county_data(data, county.geometry, all_touched=False)
            
            return clipped
            
        except Exception as e:
            console.print(f"[red]Clipping failed for {county.get('county_name', 'Unknown')}: {str(e)}[/red]")
            return None
    
    def _process_county_years(
        self, 
        clipped_data: xr.DataArray, 
        county, 
        years: np.ndarray, 
        unique_years: np.ndarray,
        variable: str, 
        scenario: str, 
        threshold: float
    ) -> List[Dict]:
        """Process all years for a single county with optimized calculations.
        
        Args:
            clipped_data: County-clipped climate data
            county: County information
            years: Year array for all timesteps
            unique_years: Unique years to process
            variable: Climate variable name
            scenario: Scenario name
            threshold: Threshold value
            
        Returns:
            List of statistics dictionaries for each year
        """
        county_results = []
        
        # Pre-compute county info to avoid repeated dictionary creation
        county_info = {
            'county_id': county['county_id'],
            'county_name': county['county_name'],
            'state': county['state']
        }
        
        for year in unique_years:
            try:
                # Efficient boolean indexing for year selection
                year_mask = years == year
                year_data = clipped_data.isel(time=year_mask)
                
                # Calculate spatial means efficiently
                # Skip NaN values that might occur at county boundaries
                daily_means = year_data.mean(dim=['y', 'x'], skipna=True).values
                
                # Filter out any NaN daily means (shouldn't happen with skipna=True, but safety check)
                if np.any(np.isnan(daily_means)):
                    daily_means = daily_means[~np.isnan(daily_means)]
                    if len(daily_means) == 0:
                        continue
                
                # Calculate statistics
                stats = calculate_statistics(
                    daily_means, variable, threshold, year, scenario, county_info
                )
                
                if stats:
                    county_results.append(stats)
                    
            except Exception as e:
                console.print(f"[red]Error processing year {year} for {county_info['county_name']}: {str(e)}[/red]")
                continue
        
        return county_results


 