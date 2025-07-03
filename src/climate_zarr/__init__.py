"""
Climate Zarr Toolkit

Interactive CLI toolkit for processing climate data with NetCDF to Zarr conversion
and county-level statistical analysis.
"""

from climate_zarr._version import __version__

__author__ = "Chris Mihiar"
__email__ = "chris.mihiar.fs@gmail.com"

# Public API exports
from climate_zarr.climate_config import (
    ClimateConfig,
    CompressionConfig,
    ChunkingConfig,
    RegionConfig,
    ProcessingConfig,
    get_config,
)
from climate_zarr.stack_nc_to_zarr import stack_netcdf_to_zarr
from climate_zarr.calculate_county_stats import ModernCountyProcessor

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Configuration
    "ClimateConfig",
    "CompressionConfig",
    "ChunkingConfig",
    "RegionConfig",
    "ProcessingConfig",
    "get_config",
    # Core functions
    "stack_netcdf_to_zarr",
    "ModernCountyProcessor",
]