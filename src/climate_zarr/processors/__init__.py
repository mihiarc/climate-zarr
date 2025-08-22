"""
Climate data processors package.

This package provides modular processors for different climate variables
and processing strategies.
"""

from .base_processor import BaseCountyProcessor
from .precipitation_processor import PrecipitationProcessor
from .temperature_processor import TemperatureProcessor
from .tasmax_processor import TasMaxProcessor
from .tasmin_processor import TasMinProcessor
from .processing_strategies import (
    VectorizedStrategy,
)

__all__ = [
    # Base processor
    "BaseCountyProcessor",
    # Variable processors
    "PrecipitationProcessor",
    "TemperatureProcessor", 
    "TasMaxProcessor",
    "TasMinProcessor",
    # Processing strategies
    "VectorizedStrategy",
] 