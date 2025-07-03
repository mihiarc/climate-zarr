#!/usr/bin/env python
"""Modern configuration management for climate data processing."""

from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import os


class CompressionConfig(BaseModel):
    """Compression configuration settings."""
    
    algorithm: str = Field(default="zstd", description="Compression algorithm")
    level: int = Field(default=5, ge=1, le=9, description="Compression level")
    shuffle: bool = Field(default=True, description="Enable byte shuffling")
    typesize: int = Field(default=4, description="Type size for optimization")
    
    @field_validator('algorithm')
    def validate_algorithm(cls, v):
        valid_algorithms = ['zstd', 'lz4', 'zlib', 'gzip', 'snappy']
        if v not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")
        return v


class ChunkingConfig(BaseModel):
    """Chunking strategy configuration."""
    
    time: int = Field(default=365, gt=0, description="Time dimension chunks")
    lat: int = Field(default=180, gt=0, description="Latitude dimension chunks")
    lon: int = Field(default=360, gt=0, description="Longitude dimension chunks")
    auto_optimize: bool = Field(default=True, description="Auto-optimize chunks based on data")
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for xarray."""
        return {
            'time': self.time,
            'lat': self.lat,
            'lon': self.lon
        }


class RegionConfig(BaseModel):
    """Geographic region configuration."""
    
    name: str = Field(description="Region name")
    lat_min: float = Field(description="Minimum latitude")
    lat_max: float = Field(description="Maximum latitude")
    lon_min: float = Field(description="Minimum longitude")
    lon_max: float = Field(description="Maximum longitude")
    
    @field_validator('lat_min', 'lat_max')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v
    
    @field_validator('lon_min', 'lon_max')
    def validate_longitude(cls, v):
        if not -180 <= v <= 360:
            raise ValueError("Longitude must be between -180 and 360")
        return v


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    
    n_workers: int = Field(default=4, ge=1, description="Number of worker processes")
    memory_limit: str = Field(default="4GB", description="Memory limit per worker")
    use_distributed: bool = Field(default=False, description="Use Dask distributed")
    chunk_by_county: bool = Field(default=True, description="Process counties in chunks")
    progress_bar: bool = Field(default=True, description="Show progress bars")


class ClimateConfig(BaseModel):
    """Main climate processing configuration."""
    
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    # Predefined regions
    regions: Dict[str, RegionConfig] = Field(default_factory=lambda: {
        'conus': RegionConfig(
            name='CONUS',
            lat_min=24.0,
            lat_max=50.0,
            lon_min=-125.0,
            lon_max=-66.0
        ),
        'alaska': RegionConfig(
            name='Alaska',
            lat_min=54.0,
            lat_max=72.0,
            lon_min=-180.0,
            lon_max=-129.0
        ),
        'hawaii': RegionConfig(
            name='Hawaii',
            lat_min=18.0,
            lat_max=23.0,
            lon_min=-162.0,
            lon_max=-154.0
        ),
        'guam': RegionConfig(
            name='Guam/MP',
            lat_min=13.0,
            lat_max=21.0,
            lon_min=144.0,
            lon_max=146.0
        ),
        'puerto_rico': RegionConfig(
            name='Puerto Rico/USVI',
            lat_min=17.5,
            lat_max=18.6,
            lon_min=-67.5,
            lon_max=-64.5
        ),
        'pr_vi': RegionConfig(
            name='Puerto Rico/USVI',
            lat_min=17.5,
            lat_max=18.6,
            lon_min=-67.5,
            lon_max=-64.5
        ),
        'global': RegionConfig(
            name='Global',
            lat_min=-90.0,
            lat_max=90.0,
            lon_min=-180.0,
            lon_max=180.0
        )
    })
    
    # File paths
    default_output_dir: Path = Field(default=Path('./output'), description="Default output directory")
    temp_dir: Optional[Path] = Field(default=None, description="Temporary directory")
    
    # Zarr format settings
    zarr_format: int = Field(default=3, ge=2, le=3, description="Zarr format version")
    zarr_fallback: bool = Field(default=True, description="Fallback to v2 if v3 fails")
    
    # Virtual store settings
    prefer_virtual: bool = Field(default=True, description="Prefer virtual stores when possible")
    kerchunk_fallback: bool = Field(default=True, description="Use Kerchunk if VirtualiZarr fails")
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_dir: Optional[Path] = Field(default=None, description="Cache directory")
    
    @field_validator('default_output_dir', 'temp_dir', 'cache_dir', mode='before')
    def validate_paths(cls, v):
        if v is not None:
            return Path(v)
        return v
    
    def get_region(self, name: str) -> RegionConfig:
        """Get region configuration by name."""
        if name.lower() not in self.regions:
            raise ValueError(f"Unknown region: {name}. Available: {list(self.regions.keys())}")
        return self.regions[name.lower()]
    
    def setup_directories(self):
        """Create necessary directories."""
        self.default_output_dir.mkdir(parents=True, exist_ok=True)
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'ClimateConfig':
        """Create config from environment variables."""
        config_data = {}
        
        # Processing settings from environment
        if workers := os.getenv('CLIMATE_WORKERS'):
            config_data.setdefault('processing', {})['n_workers'] = int(workers)
        
        if memory := os.getenv('CLIMATE_MEMORY_LIMIT'):
            config_data.setdefault('processing', {})['memory_limit'] = memory
        
        # Compression settings
        if algorithm := os.getenv('CLIMATE_COMPRESSION'):
            config_data.setdefault('compression', {})['algorithm'] = algorithm
        
        if level := os.getenv('CLIMATE_COMPRESSION_LEVEL'):
            config_data.setdefault('compression', {})['level'] = int(level)
        
        # Output directory
        if output_dir := os.getenv('CLIMATE_OUTPUT_DIR'):
            config_data['default_output_dir'] = output_dir
        
        return cls(**config_data)
    
    def save_config(self, path: Path):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
    
    @classmethod
    def load_config(cls, path: Path) -> 'ClimateConfig':
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Global configuration instance
DEFAULT_CONFIG = ClimateConfig()

def get_config() -> ClimateConfig:
    """Get the global configuration instance."""
    return DEFAULT_CONFIG

def set_config(config: ClimateConfig):
    """Set the global configuration instance."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config 