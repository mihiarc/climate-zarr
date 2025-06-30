"""Utilities for working with shapefiles and extracting county boundaries.

This module provides functions for loading county shapefiles and extracting
boundary information for use in climate data processing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


class CountyBoundsLookup:
    """Efficient lookup of county boundaries from shapefile."""
    
    def __init__(self, shapefile_path: str = None):
        """Initialize the county bounds lookup.
        
        Args:
            shapefile_path: Path to county shapefile. If None, uses default location.
        """
        if shapefile_path is None:
            shapefile_path = "/home/mihiarc/repos/claude_climate/data/shapefiles/tl_2024_us_county.shp"
        
        self.shapefile_path = Path(shapefile_path)
        self._counties_gdf = None
        self._bounds_cache = {}
        
        if not self.shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {self.shapefile_path}")
    
    @property
    def counties_gdf(self) -> gpd.GeoDataFrame:
        """Lazy load the counties GeoDataFrame."""
        if self._counties_gdf is None:
            logger.info(f"Loading counties from {self.shapefile_path}")
            self._counties_gdf = gpd.read_file(self.shapefile_path)
            # Ensure GEOID is string type
            self._counties_gdf['GEOID'] = self._counties_gdf['GEOID'].astype(str)
            logger.info(f"Loaded {len(self._counties_gdf)} counties")
        return self._counties_gdf
    
    @lru_cache(maxsize=4000)
    def get_county_bounds(self, county_id: str) -> Optional[List[float]]:
        """Get bounds for a county by GEOID.
        
        Args:
            county_id: County GEOID (e.g., '06037' for Los Angeles County)
            
        Returns:
            List of [min_lon, min_lat, max_lon, max_lat] or None if not found
        """
        # Check cache first
        if county_id in self._bounds_cache:
            return self._bounds_cache[county_id]
        
        # Find county in GeoDataFrame
        county = self.counties_gdf[self.counties_gdf['GEOID'] == county_id]
        
        if county.empty:
            logger.warning(f"County {county_id} not found in shapefile")
            return None
        
        # Get bounds from geometry
        bounds = county.geometry.iloc[0].bounds
        bounds_list = list(bounds)  # (minx, miny, maxx, maxy)
        
        # Cache for future lookups
        self._bounds_cache[county_id] = bounds_list
        
        return bounds_list
    
    def get_county_info(self, county_id: str) -> Optional[Dict[str, Any]]:
        """Get full county information including bounds and metadata.
        
        Args:
            county_id: County GEOID
            
        Returns:
            Dictionary with county information or None if not found
        """
        county = self.counties_gdf[self.counties_gdf['GEOID'] == county_id]
        
        if county.empty:
            return None
        
        county_row = county.iloc[0]
        bounds = list(county_row.geometry.bounds)
        
        return {
            'geoid': county_id,
            'name': f"{county_row.get('NAME', 'Unknown')}, {county_row.get('STUSPS', 'XX')}",
            'state': county_row.get('STUSPS', 'XX'),
            'bounds': bounds,
            'area': county_row.geometry.area,
            'centroid': [county_row.geometry.centroid.x, county_row.geometry.centroid.y]
        }
    
    def get_counties_in_bounds(self, bounds: List[float]) -> List[str]:
        """Get all county GEOIDs that intersect with given bounds.
        
        Args:
            bounds: [min_lon, min_lat, max_lon, max_lat]
            
        Returns:
            List of county GEOIDs
        """
        from shapely.geometry import box
        
        # Create bounding box geometry
        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        
        # Find intersecting counties
        intersecting = self.counties_gdf[self.counties_gdf.geometry.intersects(bbox)]
        
        return intersecting['GEOID'].tolist()
    
    def get_counties_by_state(self, state_code: str) -> List[str]:
        """Get all county GEOIDs for a given state.
        
        Args:
            state_code: Two-letter state code (e.g., 'CA')
            
        Returns:
            List of county GEOIDs
        """
        state_counties = self.counties_gdf[self.counties_gdf['STUSPS'] == state_code]
        return state_counties['GEOID'].tolist()
    
    def create_spatial_tiles(self, tile_size_degrees: float = 2.0) -> Dict[str, Dict[str, Any]]:
        """Create spatial tiles for efficient batch processing.
        
        Args:
            tile_size_degrees: Size of each tile in degrees
            
        Returns:
            Dictionary of tile_id -> {'bounds': [...], 'counties': [...]}
        """
        from shapely.geometry import box
        import numpy as np
        
        # Get overall bounds
        total_bounds = self.counties_gdf.total_bounds
        min_lon, min_lat, max_lon, max_lat = total_bounds
        
        # Create grid of tiles
        lon_steps = np.arange(min_lon, max_lon, tile_size_degrees)
        lat_steps = np.arange(min_lat, max_lat, tile_size_degrees)
        
        tiles = {}
        tile_id = 0
        
        for lon in lon_steps:
            for lat in lat_steps:
                tile_bounds = [
                    lon, lat,
                    min(lon + tile_size_degrees, max_lon),
                    min(lat + tile_size_degrees, max_lat)
                ]
                
                # Find counties in this tile
                counties = self.get_counties_in_bounds(tile_bounds)
                
                if counties:
                    tiles[f"tile_{tile_id:04d}"] = {
                        'bounds': tile_bounds,
                        'counties': counties,
                        'centroid': [
                            (tile_bounds[0] + tile_bounds[2]) / 2,
                            (tile_bounds[1] + tile_bounds[3]) / 2
                        ]
                    }
                    tile_id += 1
        
        logger.info(f"Created {len(tiles)} spatial tiles covering {len(self.counties_gdf)} counties")
        return tiles


# Convenience function for quick lookups
_default_lookup = None

def get_county_bounds(county_id: str, shapefile_path: str = None) -> Optional[List[float]]:
    """Get county bounds using the default lookup instance.
    
    Args:
        county_id: County GEOID
        shapefile_path: Optional path to shapefile (uses default if None)
        
    Returns:
        List of [min_lon, min_lat, max_lon, max_lat] or None if not found
    """
    global _default_lookup
    
    if _default_lookup is None:
        _default_lookup = CountyBoundsLookup(shapefile_path)
    
    return _default_lookup.get_county_bounds(county_id)


def get_county_info(county_id: str, shapefile_path: str = None) -> Optional[Dict[str, Any]]:
    """Get county information using the default lookup instance.
    
    Args:
        county_id: County GEOID
        shapefile_path: Optional path to shapefile (uses default if None)
        
    Returns:
        Dictionary with county information or None if not found
    """
    global _default_lookup
    
    if _default_lookup is None:
        _default_lookup = CountyBoundsLookup(shapefile_path)
    
    return _default_lookup.get_county_info(county_id)