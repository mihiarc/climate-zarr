"""
Modular processing strategies package.

This package provides a modular namespace for strategy classes. For backward
compatibility, classes are re-exported from the existing monolithic module.
"""

# Re-export from the streamlined module
from ..processing_strategies import (
    ProcessingStrategy,
    UltraFastStrategy,
)

__all__ = [
    "ProcessingStrategy",
    "UltraFastStrategy",
]

