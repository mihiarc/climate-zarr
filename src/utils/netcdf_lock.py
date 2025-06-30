"""Thread-safe NetCDF file access utilities.

NetCDF files using HDF5 backend are not thread-safe, which can cause
segmentation faults when accessed concurrently. This module provides
locking mechanisms for safe concurrent access.
"""

import threading
import multiprocessing
from pathlib import Path
from typing import Dict, Optional, Union
import os
import fcntl
import tempfile

# Global lock for NetCDF file access
_netcdf_lock = threading.Lock()

# Semaphore to limit concurrent NetCDF operations
# This prevents too many simultaneous file opens which can cause segfaults
_netcdf_semaphore = threading.Semaphore(4)  # Allow max 4 concurrent operations

# Per-file locks for finer-grained control
_file_locks: Dict[str, threading.Lock] = {}
_file_locks_lock = threading.Lock()


def get_netcdf_lock() -> threading.Lock:
    """Get the global NetCDF access lock."""
    return _netcdf_lock


def get_netcdf_semaphore() -> threading.Semaphore:
    """Get the NetCDF semaphore for limiting concurrent operations."""
    return _netcdf_semaphore


def get_file_lock(file_path: Path) -> threading.Lock:
    """Get a lock specific to a file path.
    
    Args:
        file_path: Path to the NetCDF file
        
    Returns:
        Threading lock for this specific file
    """
    # Normalize the path
    abs_path = str(file_path.absolute())
    
    # Get or create lock for this file
    with _file_locks_lock:
        if abs_path not in _file_locks:
            _file_locks[abs_path] = threading.Lock()
        return _file_locks[abs_path]


def set_netcdf_thread_safe():
    """Set environment variables for thread-safe NetCDF access."""
    # Disable HDF5 file locking which can cause issues
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    
    # Set OMP_NUM_THREADS to 1 to avoid OpenMP issues
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Disable parallel HDF5
    os.environ['HDF5_DISABLE_PARALLEL'] = '1'


# Set thread-safe environment on import
set_netcdf_thread_safe()