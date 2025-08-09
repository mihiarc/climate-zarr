#!/usr/bin/env python3
"""
Modern US County Shapefile Splitter by Region
Uses cutting-edge geospatial tools to split counties into climate regions.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from climate_config import get_config, RegionConfig
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console
import warnings

# Setup logging and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class ModernCountySplitter:
    """Modern county shapefile splitter using latest geospatial tools."""
    
    def __init__(self, shapefile_path: str, output_dir: str = "regional_counties"):
        self.shapefile_path = Path(shapefile_path)
        self.output_dir = Path(output_dir)
        self.config = get_config()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def load_counties(self) -> gpd.GeoDataFrame:
        """Load county shapefile using modern pyogrio engine."""
        logger.info(f"Loading counties from {self.shapefile_path}")
        
        # Use pyogrio engine (modern replacement for fiona)
        counties = gpd.read_file(self.shapefile_path, engine='pyogrio')
        
        logger.info(f"Loaded {len(counties)} counties")
        logger.info(f"Original CRS: {counties.crs}")
        
        # Ensure we're using WGS84 for lat/lon operations
        if counties.crs != 'EPSG:4326':
            logger.info("Reprojecting to WGS84 (EPSG:4326)")
            counties = counties.to_crs('EPSG:4326')
        
        return counties
    
    def get_county_centroids(self, counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate county centroids for region assignment with International Date Line correction."""
        logger.info("Calculating county centroids...")
        
        def get_corrected_centroid(geometry):
            """Get centroid with International Date Line correction."""
            bounds = geometry.bounds
            min_lon, min_lat, max_lon, max_lat = bounds
            
            # Check if geometry crosses International Date Line
            if max_lon - min_lon > 180:
                # Geometry crosses date line, need special handling
                # Use bounds center with correction for date line crossing
                if min_lon < 0 and max_lon > 0:
                    # Crosses from negative to positive (typical date line crossing)
                    corrected_lon = min_lon + (360 + max_lon - min_lon) / 2
                    if corrected_lon > 180:
                        corrected_lon -= 360
                else:
                    corrected_lon = (min_lon + max_lon) / 2
            else:
                # Normal geometry, use regular centroid
                centroid = geometry.centroid
                corrected_lon = centroid.x
            
            # Latitude is usually fine
            corrected_lat = (min_lat + max_lat) / 2
            
            return corrected_lon, corrected_lat
        
        # Calculate corrected centroids
        counties_with_centroids = counties.copy()
        corrected_centroids = []
        
        for _, row in counties_with_centroids.iterrows():
            lon, lat = get_corrected_centroid(row.geometry)
            corrected_centroids.append((lon, lat))
        
        counties_with_centroids['centroid_lon'] = [c[0] for c in corrected_centroids]
        counties_with_centroids['centroid_lat'] = [c[1] for c in corrected_centroids]
        
        return counties_with_centroids
    
    def assign_region(self, lat: float, lon: float) -> str:
        """Assign a region based on lat/lon coordinates."""
        # Priority order for overlapping regions
        region_priority = ['alaska', 'hawaii', 'guam', 'puerto_rico', 'conus']
        
        for region_name in region_priority:
            if region_name in self.config.regions:
                region = self.config.regions[region_name]
                
                # Special handling for Alaska to include Aleutians West
                if region_name == 'alaska':
                    # Expand latitude range to include Aleutians West (down to 51°N)
                    alaska_lat_min = 51.0  # Instead of 54.0
                    alaska_lat_max = region.lat_max
                    
                    # Handle International Date Line crossing for Aleutians West
                    # Alaska spans from -180° to -129°, but Aleutians West crosses to +179°
                    lat_in_range = alaska_lat_min <= lat <= alaska_lat_max
                    
                    # Check longitude: either in normal Alaska range OR in Aleutians West range
                    lon_in_alaska_main = region.lon_min <= lon <= region.lon_max  # -180° to -129°
                    lon_in_aleutians_west = 170.0 <= lon <= 180.0  # Eastern side of date line
                    
                    if lat_in_range and (lon_in_alaska_main or lon_in_aleutians_west):
                        return region_name
                else:
                    # Standard region checking for all other regions
                    if (region.lat_min <= lat <= region.lat_max and 
                        region.lon_min <= lon <= region.lon_max):
                        return region_name
        
        # If no region matches, assign to 'other'
        return 'other'
    
    def split_counties_by_region(self, counties: gpd.GeoDataFrame) -> Dict[str, gpd.GeoDataFrame]:
        """Split counties into regions based on centroid locations."""
        logger.info("Assigning counties to regions...")
        
        counties_with_regions = counties.copy()
        
        # Assign regions with Rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Assigning regions", total=len(counties_with_regions))
            
            regions = []
            for idx, row in counties_with_regions.iterrows():
                region = self.assign_region(row['centroid_lat'], row['centroid_lon'])
                regions.append(region)
                progress.advance(task)
            
            counties_with_regions['region'] = regions
        
        # Group by region
        regional_counties = {}
        region_counts = counties_with_regions['region'].value_counts()
        
        logger.info("Regional distribution:")
        for region, count in region_counts.items():
            logger.info(f"  {region}: {count} counties")
            regional_counties[region] = counties_with_regions[
                counties_with_regions['region'] == region
            ].copy()
        
        return regional_counties
    
    def save_regional_shapefiles(self, regional_counties: Dict[str, gpd.GeoDataFrame]) -> List[Path]:
        """Save each region as a separate shapefile."""
        logger.info("Saving regional shapefiles...")
        
        saved_files = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Saving regions", total=len(regional_counties))
            
            for region_name, gdf in regional_counties.items():
                if len(gdf) == 0:
                    logger.warning(f"No counties found for region: {region_name}")
                    progress.advance(task)
                    continue
                
                # Clean up the dataframe
                output_gdf = gdf.drop(columns=['centroid_lon', 'centroid_lat'], errors='ignore')
                
                # Create filename
                output_file = self.output_dir / f"{region_name}_counties.shp"
                
                # Save using pyogrio engine for modern performance
                output_gdf.to_file(output_file, engine='pyogrio')
                saved_files.append(output_file)
                
                logger.info(f"Saved {len(output_gdf)} counties to {output_file}")
                progress.advance(task)
        
        return saved_files
    
    def create_summary_report(self, regional_counties: Dict[str, gpd.GeoDataFrame]) -> Path:
        """Create a summary report of the regional split."""
        summary_data = []
        
        for region_name, gdf in regional_counties.items():
            if region_name in self.config.regions:
                region_config = self.config.regions[region_name]
                summary_data.append({
                    'region': region_name,
                    'display_name': region_config.name,
                    'county_count': len(gdf),
                    'lat_range': f"{region_config.lat_min} to {region_config.lat_max}",
                    'lon_range': f"{region_config.lon_min} to {region_config.lon_max}",
                    'area_km2': gdf.to_crs('EPSG:3857').geometry.area.sum() / 1e6  # Convert to km²
                })
            else:
                summary_data.append({
                    'region': region_name,
                    'display_name': 'Other/Unassigned',
                    'county_count': len(gdf),
                    'lat_range': 'N/A',
                    'lon_range': 'N/A',
                    'area_km2': gdf.to_crs('EPSG:3857').geometry.area.sum() / 1e6 if len(gdf) > 0 else 0
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / "regional_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Summary report saved to {summary_file}")
        return summary_file
    
    def run(self) -> Tuple[List[Path], Path]:
        """Execute the complete county splitting workflow."""
        logger.info("Starting modern county splitter workflow...")
        
        # Load counties
        counties = self.load_counties()
        
        # Calculate centroids
        counties_with_centroids = self.get_county_centroids(counties)
        
        # Split by regions
        regional_counties = self.split_counties_by_region(counties_with_centroids)
        
        # Save regional shapefiles
        saved_files = self.save_regional_shapefiles(regional_counties)
        
        # Create summary report
        summary_file = self.create_summary_report(regional_counties)
        
        logger.info("✅ County splitting completed successfully!")
        logger.info(f"📊 Created {len(saved_files)} regional shapefiles")
        logger.info(f"📋 Summary report: {summary_file}")
        
        return saved_files, summary_file


def main():
    """Main execution function."""
    console.print("🌍 [bold blue]Modern US County Shapefile Splitter[/bold blue]")
    console.print("=" * 50)
    
    # Initialize splitter
    splitter = ModernCountySplitter(
        shapefile_path="tl_2024_us_county/tl_2024_us_county.shp",
        output_dir="regional_counties"
    )
    
    # Run the splitting process
    saved_files, summary_file = splitter.run()
    
    console.print("\n🎉 [bold green]Process completed![/bold green]")
    console.print(f"Regional shapefiles saved to: [cyan]{splitter.output_dir}[/cyan]")
    console.print("\n[bold]Generated files:[/bold]")
    for file_path in saved_files:
        console.print(f"  📄 [green]{file_path.name}[/green]")
    console.print(f"  📊 [yellow]{summary_file.name}[/yellow]")


if __name__ == "__main__":
    main() 