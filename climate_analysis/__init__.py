"""Climate Analysis Package

A comprehensive climate data analysis package for processing NEX-GDDP data
and calculating climate indicators at the county level.
"""

__version__ = "0.1.0"

from .core.unified_processor import UnifiedParallelProcessor
from .core.unified_calculator import UnifiedClimateCalculator

__all__ = ["UnifiedParallelProcessor", "UnifiedClimateCalculator"]