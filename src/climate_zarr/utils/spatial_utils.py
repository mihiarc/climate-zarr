#!/usr/bin/env python
"""Spatial processing utilities for climate data."""

from typing import List
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import Point
from rich.console import Console

console = Console()


def normalize_longitude_coordinates(gdf, data_lon_range):
    """
    Normalize longitude coordinates to handle antimeridian crossing.
    
    Args:
        gdf: GeoDataFrame with county geometries
        data_lon_range: Tuple of (min_lon, max_lon) from climate data
        
    Returns:
        GeoDataFrame with normalized longitude coordinates
    """
    import shapely.ops
    from shapely.geometry import mapping, shape
    
    data_min_lon, data_max_lon = data_lon_range
    
    # Check if shapefile crosses antimeridian (has both negative and positive longitudes)
    shapefile_bounds = gdf.total_bounds
    crosses_antimeridian = shapefile_bounds[0] < 0 and shapefile_bounds[2] > 0
    
    # If data is entirely in negative longitude space and shapefile crosses antimeridian
    if data_max_lon < 0 and crosses_antimeridian:
        console.print(f"[yellow]Normalizing longitude coordinates for antimeridian crossing...[/yellow]")
        console.print(f"[yellow]Data range: {data_min_lon:.3f} to {data_max_lon:.3f}[/yellow]")
        console.print(f"[yellow]Shapefile bounds: {shapefile_bounds[0]:.3f} to {shapefile_bounds[2]:.3f}[/yellow]")
        
        def normalize_geometry(geom):
            """Convert positive longitudes to negative (-180 to 0) range for antimeridian crossing."""
            def normalize_coords(coords):
                normalized = []
                for lon, lat in coords:
                    # Convert positive longitudes (typically 0-180) to negative (-180 to 0)
                    if lon > 0:
                        new_lon = lon - 360
                    else:
                        new_lon = lon
                    normalized.append((new_lon, lat))
                return normalized
            
            if geom.geom_type == 'Polygon':
                exterior_coords = normalize_coords(list(geom.exterior.coords))
                interiors = [normalize_coords(list(interior.coords)) for interior in geom.interiors]
                return shape({
                    'type': 'Polygon',
                    'coordinates': [exterior_coords] + interiors
                })
            elif geom.geom_type == 'MultiPolygon':
                new_polygons = []
                for polygon in geom.geoms:
                    exterior_coords = normalize_coords(list(polygon.exterior.coords))
                    interiors = [normalize_coords(list(interior.coords)) for interior in polygon.interiors]
                    new_polygons.append([exterior_coords] + interiors)
                return shape({
                    'type': 'MultiPolygon',
                    'coordinates': new_polygons
                })
            else:
                return geom
        
        # Apply normalization to all geometries
        gdf_normalized = gdf.copy()
        gdf_normalized['geometry'] = gdf_normalized['geometry'].apply(normalize_geometry)
        
        new_bounds = gdf_normalized.total_bounds
        console.print(f"[green]Normalized bounds: {new_bounds[0]:.3f} to {new_bounds[2]:.3f}[/green]")
        
        return gdf_normalized
    
    return gdf


def interpolate_from_nearest_cells(county_geometry, climate_data, lats, lons, max_distance_degrees=0.25, n_nearest=4):
    """
    Interpolate climate data for a county using inverse distance weighting from nearby grid cells.
    
    Only uses grid cells within a reasonable distance to maintain realistic gradients.
    
    Args:
        county_geometry: Shapely geometry of the county
        climate_data: 2D numpy array of climate data (lat, lon)
        lats: 1D array of latitude coordinates
        lons: 1D array of longitude coordinates
        max_distance_degrees: Maximum distance in degrees to consider grid cells (default: 0.25° ≈ 28km)
        n_nearest: Maximum number of nearest cells to use for interpolation
    
    Returns:
        Interpolated climate value
    """
    # Get county centroid
    centroid = county_geometry.centroid
    
    # Create arrays of all grid cell coordinates
    lat_coords, lon_coords = np.meshgrid(lats, lons, indexing='ij')
    
    # Calculate distances from centroid to all grid cells
    distances = np.sqrt((lat_coords - centroid.y)**2 + (lon_coords - centroid.x)**2)
    
    # First, filter to only nearby cells within max_distance
    nearby_mask = distances <= max_distance_degrees
    
    if not np.any(nearby_mask):
        # If no cells within max_distance, return NaN - don't interpolate from far cells
        return np.nan
    
    # Get distances and indices for nearby cells only
    nearby_distances = distances[nearby_mask]
    nearby_climate_data = climate_data[nearby_mask]
    
    # Sort by distance and take up to n_nearest
    sorted_indices = np.argsort(nearby_distances)[:n_nearest]
    
    # Extract values from nearest nearby cells
    values = []
    weights = []
    
    for idx in sorted_indices:
        value = nearby_climate_data[idx]
        distance = nearby_distances[idx]
        
        if not np.isnan(value):
            values.append(value)
            # Use inverse distance weighting with minimum distance to avoid division by zero
            weights.append(1.0 / (distance + 1e-6))
    
    if len(values) == 0:
        return np.nan
    
    # Calculate weighted average
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    weight_sum = sum(weights)
    
    return weighted_sum / weight_sum


def get_radius_average(county_geometry, climate_data, lats, lons, radius_degrees=0.2):
    """
    Get average of climate grid cells within a radius of the county centroid.
    
    Args:
        county_geometry: Shapely geometry of the county
        climate_data: 2D numpy array of climate data (lat, lon)
        lats: 1D array of latitude coordinates
        lons: 1D array of longitude coordinates
        radius_degrees: Search radius in degrees (default: 0.2° ≈ 22km)
    
    Returns:
        Average climate value within radius
    """
    # Get county centroid
    centroid = county_geometry.centroid
    
    # Create coordinate grids
    lat_coords, lon_coords = np.meshgrid(lats, lons, indexing='ij')
    
    # Calculate distances from centroid to all grid points
    distances = np.sqrt((lat_coords - centroid.y)**2 + (lon_coords - centroid.x)**2)
    
    # Find points within radius
    within_radius = distances <= radius_degrees
    
    # Get climate data within radius
    nearby_data = climate_data[within_radius]
    
    # Remove NaN values and calculate average
    valid_data = nearby_data[~np.isnan(nearby_data)]
    
    if len(valid_data) > 0:
        return np.mean(valid_data)
    else:
        return np.nan


def get_nearest_cell_value(county_geometry, climate_data, lats, lons):
    """
    Get climate value from the single nearest grid cell to county centroid.
    
    Args:
        county_geometry: Shapely geometry of the county
        climate_data: 2D numpy array of climate data (lat, lon)
        lats: 1D array of latitude coordinates
        lons: 1D array of longitude coordinates
    
    Returns:
        Climate value from nearest cell
    """
    # Get county centroid
    centroid = county_geometry.centroid
    
    # Find nearest grid cell
    lat_idx = np.argmin(np.abs(lats - centroid.y))
    lon_idx = np.argmin(np.abs(lons - centroid.x))
    
    return climate_data[lat_idx, lon_idx]


def extract_county_climate_data_with_interpolation(
    county_row, 
    county_raster, 
    climate_data, 
    lats, 
    lons
):
    """
    Extract climate data for a county using rasterization with interpolation fallback.
    
    Args:
        county_row: County row from GeoDataFrame with geometry and raster_id
        county_raster: County raster mask
        climate_data: 2D numpy array of climate data (lat, lon)
        lats: 1D array of latitude coordinates
        lons: 1D array of longitude coordinates
    
    Returns:
        Tuple of (climate_value, extraction_method)
    """
    raster_id = county_row['raster_id']
    
    # Method 1: Try standard rasterization
    county_mask = county_raster == raster_id
    
    if np.any(county_mask):
        county_data = climate_data[county_mask]
        
        if county_data.size > 0:
            valid_data = county_data[~np.isnan(county_data)]
            
            if len(valid_data) > 0:
                return np.mean(valid_data), "rasterization"
    
    # Method 2: Try inverse distance weighting interpolation (only nearby cells)
    idw_value = interpolate_from_nearest_cells(
        county_row.geometry, climate_data, lats, lons, max_distance_degrees=0.25, n_nearest=4
    )
    
    if not np.isnan(idw_value):
        return idw_value, "idw_interpolation"
    
    # Method 3: Try radius averaging (smaller radius for realistic gradients)
    radius_value = get_radius_average(
        county_row.geometry, climate_data, lats, lons, radius_degrees=0.2
    )
    
    if not np.isnan(radius_value):
        return radius_value, "radius_average"
    
    # Method 4: Try nearest neighbor (last resort)
    nearest_value = get_nearest_cell_value(
        county_row.geometry, climate_data, lats, lons
    )
    
    if not np.isnan(nearest_value):
        return nearest_value, "nearest_neighbor"
    
    # If all methods fail, return NaN
    return np.nan, "no_data_available"


def create_county_raster_with_ids(
    gdf: gpd.GeoDataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
    all_touched: bool = True
) -> tuple[np.ndarray, set]:
    """Create a raster mask for counties and return mask and rasterized id set.

    This retains the extended API used internally (mask, rasterized_ids).
    """
    console.print("[cyan]Creating county raster mask...[/cyan]")
    
    # Create transform for the zarr grid
    transform = from_bounds(
        lons.min(), lats.min(), lons.max(), lats.max(),
        len(lons), len(lats)
    )
    
    # Add unique IDs to counties for rasterization
    gdf_with_ids = gdf.copy()
    if 'raster_id' not in gdf_with_ids.columns:
        gdf_with_ids['raster_id'] = range(1, len(gdf) + 1)
    
    # Initialize the county raster
    county_raster = np.zeros((len(lats), len(lons)), dtype='uint16')
    
    # Batch rasterization to avoid pixel conflicts
    batch_size = 100  # Process counties in smaller batches
    rasterized_ids = set()
    
    console.print(f"[cyan]Processing {len(gdf_with_ids)} counties in batches of {batch_size}...[/cyan]")
    
    for i in range(0, len(gdf_with_ids), batch_size):
        batch_gdf = gdf_with_ids.iloc[i:i+batch_size]
        
        # Create shapes for this batch
        batch_shapes = [
            (geom, raster_id)
            for geom, raster_id in zip(batch_gdf.geometry, batch_gdf.raster_id)
            if geom is not None and not getattr(geom, "is_empty", False)
        ]
        
        # Rasterize this batch
        batch_raster = rasterize(
            batch_shapes,
            out_shape=(len(lats), len(lons)),
            transform=transform,
            fill=0,
            dtype='uint16',
            all_touched=all_touched
        )
        
        # Merge with main raster, avoiding conflicts
        # Priority: first-come-first-served for pixels
        mask = (county_raster == 0) & (batch_raster > 0)
        county_raster[mask] = batch_raster[mask]
        
        # Track which counties were successfully rasterized in this batch
        batch_rasterized = set(np.unique(batch_raster[batch_raster > 0]))
        rasterized_ids.update(batch_rasterized)
    
    # Check for any remaining missing counties and apply centroid fallback
    all_ids = set(gdf_with_ids['raster_id'])
    missing_ids = all_ids - rasterized_ids
    
    console.print(f"[cyan]Batch rasterization: {len(rasterized_ids)} counties[/cyan]")
    
    # Centroid-based assignment for any remaining missing counties
    if missing_ids:
        console.print(f"[yellow]Applying centroid-based assignment for {len(missing_ids)} remaining counties...[/yellow]")
        
        for raster_id in missing_ids:
            county_row = gdf_with_ids[gdf_with_ids['raster_id'] == raster_id].iloc[0]
            geom = county_row.geometry
            if geom is None or getattr(geom, "is_empty", False):
                continue
            # Get county centroid
            centroid = geom.centroid
            # Find nearest grid cell to centroid
            if (
                lons.min() <= centroid.x <= lons.max()
                and lats.min() <= centroid.y <= lats.max()
            ):
                # Find closest grid cell
                lon_idx = np.argmin(np.abs(lons - centroid.x))
                lat_idx = np.argmin(np.abs(lats - centroid.y))
                # Assign county to this pixel if it's not already assigned
                if county_raster[lat_idx, lon_idx] == 0:
                    county_raster[lat_idx, lon_idx] = raster_id
                    rasterized_ids.add(raster_id)
    
    final_missing = all_ids - rasterized_ids
    
    console.print(f"[cyan]County raster created with {len(rasterized_ids)} counties[/cyan]")
    if final_missing:
        console.print(f"[red]{len(final_missing)} counties could not be rasterized (truly outside grid bounds)[/red]")
    
    return county_raster, rasterized_ids


def create_county_raster(
    gdf: gpd.GeoDataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
    all_touched: bool = True
) -> np.ndarray:
    """Create a raster mask for counties; return only the mask (np.ndarray).

    This matches test expectations and external API usage.
    """
    raster, _ids = create_county_raster_with_ids(gdf, lats, lons, all_touched=all_touched)
    return raster


def create_comprehensive_county_results(
    gdf: gpd.GeoDataFrame,
    county_raster: np.ndarray,
    rasterized_ids: set,
    all_data: np.ndarray,
    years: np.ndarray,
    unique_years: np.ndarray,
    variable: str,
    scenario: str,
    threshold: float
) -> list:
    """Create comprehensive results ensuring all counties are included.
    
    Args:
        gdf: GeoDataFrame with county information
        county_raster: County raster mask
        rasterized_ids: Set of successfully rasterized county IDs
        all_data: Climate data array (time, lat, lon)
        years: Years array for each time step
        unique_years: Unique years in the dataset
        variable: Climate variable name
        scenario: Scenario name
        threshold: Threshold value
        
    Returns:
        List of result dictionaries for all counties
    """
    from ..utils.data_utils import calculate_statistics
    
    results = []
    
    console.print(f"[cyan]Processing {len(gdf)} counties ({len(rasterized_ids)} rasterized)[/cyan]")
    
    for idx, county_row in gdf.iterrows():
        raster_id = county_row['raster_id']
        county_info = {
            'county_id': county_row['county_id'],
            'county_name': county_row['county_name'],
            'state': county_row['state']
        }
        
        if raster_id in rasterized_ids:
            # Process counties with climate data
            county_mask = county_raster == raster_id
            
            if np.any(county_mask):
                county_data = all_data[:, county_mask]
                
                if county_data.size > 0:
                    # Use nanmean to handle mixed NaN/valid pixels
                    daily_means = np.nanmean(county_data, axis=1)
                    
                    # Process each year
                    for year in unique_years:
                        year_mask = years == year
                        year_data = daily_means[year_mask]
                        
                        stats = calculate_statistics(
                            year_data, variable, threshold, year, scenario, county_info
                        )
                        
                        if stats:
                            results.append(stats)
                        else:
                            # Create no-data record for years with no valid data
                            no_data_record = create_no_data_record(
                                county_info, year, scenario, variable, "no_valid_climate_data"
                            )
                            results.append(no_data_record)
                else:
                    # Create no-data records for all years
                    for year in unique_years:
                        no_data_record = create_no_data_record(
                            county_info, year, scenario, variable, "no_data_extracted"
                        )
                        results.append(no_data_record)
            else:
                # County was rasterized but has no pixels in final raster (conflict resolution)
                # This means it was assigned to a pixel that another county claimed
                # We should still try to get data using centroid-based extraction
                centroid = county_row.geometry.centroid
                
                # Find nearest grid cell to centroid
                if (all_data.shape[2] > 0 and all_data.shape[1] > 0):
                    # Get coordinate arrays from the data shape
                    lats = np.linspace(county_raster.shape[0]-1, 0, county_raster.shape[0])
                    lons = np.linspace(0, county_raster.shape[1]-1, county_raster.shape[1])
                    
                    # This is a fallback - create no-data records
                    for year in unique_years:
                        no_data_record = create_no_data_record(
                            county_info, year, scenario, variable, "pixel_conflict_resolved"
                        )
                        results.append(no_data_record)
                else:
                    # Create no-data records for all years
                    for year in unique_years:
                        no_data_record = create_no_data_record(
                            county_info, year, scenario, variable, "no_data_extracted"
                        )
                        results.append(no_data_record)
        else:
            # Create no-data records for counties that couldn't be rasterized
            for year in unique_years:
                no_data_record = create_no_data_record(
                    county_info, year, scenario, variable, "outside_climate_domain"
                )
                results.append(no_data_record)
    
    return results


def create_no_data_record(
    county_info: dict,
    year: int,
    scenario: str,
    variable: str,
    reason: str
) -> dict:
    """Create a no-data record for a county-year combination.
    
    Args:
        county_info: Dictionary with county information
        year: Year
        scenario: Scenario name
        variable: Variable name
        reason: Reason for no data
        
    Returns:
        Dictionary with no-data record
    """
    if variable.lower() == 'pr':
        return {
            'county_id': county_info['county_id'],
            'county_name': county_info['county_name'],
            'state': county_info['state'],
            'year': year,
            'scenario': scenario,
            'mean_daily_precip_mm': np.nan,
            'total_annual_precip_mm': np.nan,
            'days_above_threshold': 0,
            'max_daily_precip_mm': np.nan,
            'precip_95th_percentile_mm': np.nan,
            'data_status': reason
        }
    else:
        # Generic template for other variables
        return {
            'county_id': county_info['county_id'],
            'county_name': county_info['county_name'],
            'state': county_info['state'],
            'year': year,
            'scenario': scenario,
            'data_status': reason
        }


def get_time_information(data: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Extract time information from xarray DataArray.
    
    Args:
        data: xarray DataArray with time dimension
        
    Returns:
        Tuple of (years array, unique years array)
    """
    time_values = data.time.values
    
    if hasattr(time_values[0], 'year'):
        years = np.array([t.year for t in time_values])
    else:
        years = pd.to_datetime(time_values).year.values
    
    unique_years = np.unique(years)
    
    return years, unique_years


def get_coordinate_arrays(data: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Extract coordinate arrays from xarray DataArray.
    
    Args:
        data: xarray DataArray with spatial coordinates
        
    Returns:
        Tuple of (lats, lons) coordinate arrays
    """
    # Handle different coordinate names and unlabeled scenarios
    if 'lat' in data.coords:
        lats = data.lat.values
        lons = data.lon.values
    elif 'y' in data.coords:
        lats = data.y.values
        lons = data.x.values
    elif 'y' in data.dims and 'x' in data.dims:
        # Handle unlabeled dimensions - reconstruct coordinates from data shape
        console.print("[yellow]Reconstructing coordinates from unlabeled dimensions[/yellow]")
        y_size = data.sizes['y']
        x_size = data.sizes['x']
        
        # Create reasonable coordinate arrays based on typical climate data ranges
        # Assume global coverage for now - this could be refined based on actual data bounds
        lats = np.linspace(90, -90, y_size)
        lons = np.linspace(-180, 180, x_size)
        
        # Assign coordinates back to the data for future operations
        data = data.assign_coords(y=lats, x=lons)
        
        return lats, lons
    elif 'lat' in data.dims and 'lon' in data.dims:
        # Handle unlabeled lat/lon dimensions
        console.print("[yellow]Reconstructing lat/lon coordinates from unlabeled dimensions[/yellow]")
        lat_size = data.sizes['lat']
        lon_size = data.sizes['lon']
        
        lats = np.linspace(90, -90, lat_size)
        lons = np.linspace(-180, 180, lon_size)
        
        # Assign coordinates back to the data
        data = data.assign_coords(lat=lats, lon=lons)
        
        return lats, lons
    else:
        coord_names = list(data.coords)
        dim_names = list(data.dims)
        raise ValueError(
            f"Could not find spatial coordinates. "
            f"Available coordinates: {coord_names}, dimensions: {dim_names}"
        )
    
    return lats, lons


def clip_county_data(
    data: xr.DataArray,
    county_geometry,
    all_touched: bool = True
) -> xr.DataArray:
    """Clip data to a county geometry using rioxarray with coordinate preservation.
    
    Args:
        data: xarray DataArray with spatial coordinates
        county_geometry: Shapely geometry for the county
        all_touched: Whether to include all touched pixels
        
    Returns:
        Clipped DataArray with preserved coordinate labels
    """
    # Ensure spatial dims are named as expected by rioxarray (x/y)
    # Handle common cases where dims are lat/lon
    try:
        dims_set = set(data.dims)
        if 'lon' in dims_set or 'lat' in dims_set:
            rename_map = {}
            if 'lon' in dims_set:
                rename_map['lon'] = 'x'
            if 'lat' in dims_set:
                rename_map['lat'] = 'y'
            if rename_map:
                data = data.rename(rename_map)
        # If coordinates exist with lon/lat, align them as well
        coord_keys = list(data.coords)
        coord_renames = {}
        if 'lon' in coord_keys:
            coord_renames['lon'] = 'x'
        if 'lat' in coord_keys:
            coord_renames['lat'] = 'y'
        if coord_renames:
            data = data.rename(coord_renames)
        # Prefer a canonical dim order for safety
        desired_order = [d for d in ('time', 'y', 'x') if d in data.dims]
        if list(data.dims) != desired_order and desired_order:
            data = data.transpose(*desired_order)
    except Exception:
        # Best-effort standardization; proceed even if rename/transposes fail
        pass

    # Ensure data has proper spatial reference before clipping
    if not hasattr(data, 'rio') or data.rio.crs is None:
        console.print("[yellow]Setting spatial reference before clipping[/yellow]")
        data = data.rio.write_crs('EPSG:4326')
    
    # Preserve original coordinate information
    original_coords = dict(data.coords)
    original_dims = data.dims
    
    # Perform the clip operation; let no-overlap propagate as an exception
    clipped = data.rio.clip([county_geometry], all_touched=all_touched)

    # Check if coordinates were lost during clipping
    missing_coords = []
    for dim in clipped.dims:
        if dim not in clipped.coords and dim in original_coords:
            missing_coords.append(dim)

    if missing_coords:
        console.print(f"[yellow]Restoring coordinates lost during clipping: {missing_coords}[/yellow]")
        # Restore missing coordinates by reconstructing them based on the clipped data shape
        coords_to_restore = {}
        for dim in missing_coords:
            if dim in original_coords:
                orig_coord = original_coords[dim]
                new_size = clipped.sizes[dim]
                if hasattr(orig_coord, 'values'):
                    orig_values = orig_coord.values
                    if len(orig_values) >= new_size:
                        coords_to_restore[dim] = orig_values[:new_size]
                    else:
                        if dim in ['y', 'lat']:
                            coords_to_restore[dim] = np.linspace(90, -90, new_size)
                        elif dim in ['x', 'lon']:
                            coords_to_restore[dim] = np.linspace(-180, 180, new_size)
                        else:
                            coords_to_restore[dim] = np.arange(new_size)
        if coords_to_restore:
            clipped = clipped.assign_coords(coords_to_restore)

    return clipped


def get_spatial_dims(data: xr.DataArray) -> List[str]:
    """Get spatial dimension names, handling both labeled and unlabeled scenarios.
    
    Args:
        data: xarray DataArray
        
    Returns:
        List of spatial dimension names
    """
    # Check for labeled coordinates first
    if 'y' in data.coords and 'x' in data.coords:
        return ['y', 'x']
    elif 'lat' in data.coords and 'lon' in data.coords:
        return ['lat', 'lon']
    
    # Handle unlabeled dimensions by checking dimension names
    dims = list(data.dims)
    if 'y' in dims and 'x' in dims:
        return ['y', 'x']
    elif 'lat' in dims and 'lon' in dims:
        return ['lat', 'lon']
    
    # Last resort: assume the last two dimensions are spatial
    if len(dims) >= 2:
        spatial_dims = dims[-2:]  # Usually (lat, lon) or (y, x)
        console.print(f"[yellow]Using last 2 dimensions as spatial: {spatial_dims}[/yellow]")
        return spatial_dims
    
    # If we can't determine spatial dimensions, return empty list
    console.print("[red]Warning: Could not determine spatial dimensions[/red]")
    return []


def standardize_for_clipping(data: xr.DataArray) -> xr.DataArray:
    """Prepare data for rioxarray clipping operations.
    
    Args:
        data: Input xarray DataArray
        
    Returns:
        DataArray standardized for clipping
    """
    try:
        # Rename dims first if needed
        dims_set = set(data.dims)
        if 'lon' in dims_set or 'lat' in dims_set:
            rename_map = {}
            if 'lon' in dims_set:
                rename_map['lon'] = 'x'
            if 'lat' in dims_set:
                rename_map['lat'] = 'y'
            if rename_map:
                data = data.rename(rename_map)
        
        # Rename coords if present
        coord_keys = list(data.coords)
        coord_renames = {}
        if 'lon' in coord_keys:
            coord_renames['lon'] = 'x'
        if 'lat' in coord_keys:
            coord_renames['lat'] = 'y'
        if coord_renames:
            data = data.rename(coord_renames)
        
        # Ensure CRS for .rio
        try:
            import rioxarray  # noqa: F401
        except Exception:
            pass
        
        if not hasattr(data, 'rio') or data.rio.crs is None:
            data = data.rio.write_crs('EPSG:4326')
        
        # Handle longitude wrapping
        if 'x' in data.coords and float(data.x.max()) > 180:
            data = data.assign_coords(x=(data.x + 180) % 360 - 180).sortby('x')
            
    except Exception:
        pass
    
    return data