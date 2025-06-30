"""Utility modules for climate analysis."""

from .climate_utils import (
    create_annual_record,
    adjust_longitude_bounds,
    calculate_area_weighted_mean,
    format_results_for_county,
    validate_county_data,
    get_percentile_doy_index,
    celsius_to_fahrenheit,
    mm_to_inches
)

from .file_operations import (
    ensure_directory,
    save_json,
    load_json,
    load_netcdf,
    save_netcdf,
    load_pickle,
    save_pickle,
    find_files,
    retry_on_failure,
    load_shapefile,
    get_file_hash,
    get_file_info,
    clean_old_files
)

from .regional_config import (
    REGIONAL_CONFIG,
    get_region_for_state,
    needs_dateline_handling,
    adjust_bounds_for_dateline,
    get_climate_expectations,
    validate_climate_data,
    get_recommended_processing_config
)

__all__ = [
    # climate_utils
    "create_annual_record",
    "adjust_longitude_bounds",
    "calculate_area_weighted_mean",
    "format_results_for_county",
    "validate_county_data",
    "get_percentile_doy_index",
    "celsius_to_fahrenheit",
    "mm_to_inches",
    # file_operations
    "ensure_directory",
    "save_json",
    "load_json",
    "load_netcdf",
    "save_netcdf",
    "load_pickle",
    "save_pickle",
    "find_files",
    "retry_on_failure",
    "load_shapefile",
    "get_file_hash",
    "get_file_info",
    "clean_old_files",
    # regional_config
    "REGIONAL_CONFIG",
    "get_region_for_state",
    "needs_dateline_handling",
    "adjust_bounds_for_dateline",
    "get_climate_expectations",
    "validate_climate_data",
    "get_recommended_processing_config"
]