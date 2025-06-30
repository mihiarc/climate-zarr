"""Robust file I/O operations for climate data processing.

This module provides consistent error handling, retry logic, and
file management utilities used across the climate processing pipeline.
"""

import os
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List, Union, Callable
from functools import wraps
import time
import xarray as xr
import pandas as pd
import geopandas as gpd

from .netcdf_lock import get_netcdf_lock, get_file_lock, get_netcdf_semaphore

logger = logging.getLogger(__name__)


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry file operations on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            raise last_exception
        return wrapper
    return decorator


@retry_on_failure(max_attempts=3)
def load_netcdf(
    file_path: Union[str, Path],
    preselect_bounds: Optional[Dict[str, slice]] = None,
    chunks: Optional[Dict[str, int]] = None
) -> Optional[xr.Dataset]:
    """Load NetCDF file with error handling and optional pre-selection.
    
    Args:
        file_path: Path to NetCDF file
        preselect_bounds: Dictionary of dimension slices for pre-selection
        chunks: Chunking specification for dask
        
    Returns:
        xarray Dataset or None if loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        # Use semaphore to limit concurrent NetCDF operations
        with get_netcdf_semaphore():
            # Use file-specific lock to prevent concurrent access issues
            with get_file_lock(file_path):
                # Open with pre-selection if bounds provided
                if preselect_bounds:
                    ds = xr.open_dataset(file_path, chunks=chunks)
                    ds = ds.sel(**preselect_bounds)
                else:
                    ds = xr.open_dataset(file_path, chunks=chunks)
                
                # Load data into memory to avoid keeping file handle open
                # This prevents issues with concurrent access
                ds = ds.load()
        
        return ds
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def save_netcdf(
    data: xr.Dataset,
    file_path: Union[str, Path],
    compression: Optional[Dict[str, Any]] = None,
    create_parent: bool = True
) -> bool:
    """Save xarray Dataset to NetCDF with compression and error handling.
    
    Args:
        data: xarray Dataset to save
        file_path: Output file path
        compression: Compression options for to_netcdf
        create_parent: Whether to create parent directories
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    if create_parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default compression
    if compression is None:
        compression = {
            'zlib': True,
            'complevel': 4,
            'shuffle': True
        }
    
    try:
        encoding = {var: compression for var in data.data_vars}
        data.to_netcdf(file_path, encoding=encoding)
        logger.info(f"Saved {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving {file_path}: {e}")
        return False


def load_shapefile(
    file_path: Union[str, Path],
    filters: Optional[Dict[str, Any]] = None
) -> Optional[gpd.GeoDataFrame]:
    """Load shapefile with optional filtering.
    
    Args:
        file_path: Path to shapefile
        filters: Dictionary of column filters to apply
        
    Returns:
        GeoDataFrame or None if loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"Shapefile not found: {file_path}")
        return None
    
    try:
        gdf = gpd.read_file(file_path)
        
        # Apply filters if provided
        if filters:
            for column, value in filters.items():
                if column in gdf.columns:
                    if isinstance(value, list):
                        gdf = gdf[gdf[column].isin(value)]
                    else:
                        gdf = gdf[gdf[column] == value]
        
        return gdf
        
    except Exception as e:
        logger.error(f"Error loading shapefile {file_path}: {e}")
        return None


def save_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    create_parent: bool = True
) -> bool:
    """Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        file_path: Output file path
        indent: JSON indentation
        create_parent: Whether to create parent directories
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    if create_parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Any]:
    """Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data or None if loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def save_pickle(
    data: Any,
    file_path: Union[str, Path],
    create_parent: bool = True
) -> bool:
    """Save data to pickle file.
    
    Args:
        data: Data to pickle
        file_path: Output file path
        create_parent: Whether to create parent directories
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    if create_parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
        
    except Exception as e:
        logger.error(f"Error saving pickle to {file_path}: {e}")
        return False


def load_pickle(file_path: Union[str, Path]) -> Optional[Any]:
    """Load data from pickle file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded data or None if loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
            
    except Exception as e:
        logger.error(f"Error loading pickle from {file_path}: {e}")
        return None


def get_file_hash(file_path: Union[str, Path], chunk_size: int = 8192) -> Optional[str]:
    """Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read
        
    Returns:
        MD5 hash string or None if file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    hash_md5 = hashlib.md5()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None


def find_files(
    directory: Union[str, Path],
    pattern: str,
    recursive: bool = True
) -> List[Path]:
    """Find files matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        return sorted(directory.rglob(pattern))
    else:
        return sorted(directory.glob(pattern))


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_info(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Get information about a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information or None if file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    stat = file_path.stat()
    
    return {
        'path': str(file_path),
        'name': file_path.name,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified_time': stat.st_mtime,
        'created_time': stat.st_ctime,
        'is_file': file_path.is_file(),
        'is_dir': file_path.is_dir()
    }


def clean_old_files(
    directory: Union[str, Path],
    pattern: str,
    days_old: int,
    dry_run: bool = True
) -> List[Path]:
    """Clean files older than specified days.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        days_old: Age threshold in days
        dry_run: If True, only report what would be deleted
        
    Returns:
        List of deleted (or would-be deleted) files
    """
    directory = Path(directory)
    current_time = time.time()
    age_threshold = days_old * 24 * 60 * 60
    
    deleted_files = []
    
    for file_path in find_files(directory, pattern):
        file_age = current_time - file_path.stat().st_mtime
        
        if file_age > age_threshold:
            deleted_files.append(file_path)
            
            if not dry_run:
                try:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
            else:
                logger.info(f"Would delete: {file_path}")
    
    return deleted_files