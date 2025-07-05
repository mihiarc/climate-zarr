# Modular Testing Guide for Climate-Zarr

This guide describes the comprehensive testing suite for the refactored modular climate-zarr architecture.

## Overview

The testing suite is organized into several categories to ensure thorough validation of the modular system:

1. **Unit Tests** - Test individual components in isolation
2. **Integration Tests** - Test components working together
3. **End-to-End Tests** - Test complete workflows
4. **Backward Compatibility Tests** - Ensure API compatibility
5. **Performance Tests** - Validate performance characteristics

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and pytest configuration
├── run_modular_tests.py          # Test runner script
├── MODULAR_TESTING_GUIDE.md      # This guide
│
├── Unit Tests/
│   ├── test_processors.py        # Test individual processor modules
│   ├── test_strategies.py        # Test processing strategies
│   └── test_utils.py             # Test utility functions
│
├── Integration Tests/
│   ├── test_modular_integration.py    # Test module interactions
│   └── test_end_to_end_modular.py     # Full workflow tests
│
└── Compatibility Tests/
    └── test_backward_compatibility.py  # API compatibility tests
```

## Test Categories

### 1. Unit Tests

#### `test_processors.py`
Tests individual processor modules:
- **BaseProcessor**: Core functionality, column standardization, strategy selection
- **PrecipitationProcessor**: Precipitation-specific processing logic
- **TemperatureProcessor**: Temperature processing logic
- **TasMaxProcessor**: Daily maximum temperature processing
- **TasMinProcessor**: Daily minimum temperature processing
- **Error Handling**: Invalid inputs, missing data, edge cases
- **Performance**: Memory usage, processing speed

#### `test_strategies.py`
Tests processing strategies:
- **VectorizedStrategy**: Rioxarray-based clipping approach
- **UltraFastStrategy**: Memory-optimized bulk processing
- **Strategy Comparison**: Results consistency between strategies
- **Error Handling**: Invalid variables, spatial mismatches
- **Data Types**: Different coordinate systems, time frequencies

#### `test_utils.py`
Tests utility functions:
- **Unit Conversion**: Precipitation (kg/m²/s ↔ mm/day), Temperature (K ↔ °C)
- **Time Information**: Multi-year data, different frequencies
- **County Raster**: Spatial overlap detection, rasterization
- **Statistics Calculation**: All climate variables, edge cases
- **Spatial Utilities**: Clipping, masking, area-weighted means
- **Error Handling**: Invalid inputs, empty data, no overlap

### 2. Integration Tests

#### `test_modular_integration.py`
Tests module interactions:
- **Utility Integration**: Unit conversion with real data
- **Strategy Integration**: Processing with actual spatial data
- **Processor Integration**: Variable-specific processing workflows
- **ModernCountyProcessor**: Unified interface functionality
- **Backward Compatibility**: Import compatibility, API consistency

#### `test_end_to_end_modular.py`
Tests complete workflows:
- **Full Precipitation Workflow**: Shapefile → Zarr → CSV results
- **Full Temperature Workflow**: Multiple variables, different strategies
- **Multiple Variables**: Sequential processing, data consistency
- **Strategy Selection**: Automatic selection based on data size
- **Error Handling**: File I/O errors, invalid configurations
- **Performance**: Memory usage, processing time, scalability

### 3. Compatibility Tests

#### `test_backward_compatibility.py`
Ensures API compatibility:
- **Import Compatibility**: All expected imports work
- **Method Signatures**: Parameter names and defaults unchanged
- **Data Structures**: Output format consistency
- **Configuration**: Worker count, thresholds, processing options
- **Error Messages**: Consistent error handling

## Running Tests

### Quick Start

```bash
# Run all modular tests
python tests/run_modular_tests.py

# Run specific test file
python tests/run_modular_tests.py test_processors

# Run with pytest directly
pytest tests/test_processors.py -v
```

### Test Categories

```bash
# Run only unit tests
pytest -m unit tests/

# Run only integration tests  
pytest -m integration tests/

# Run slow tests (end-to-end)
pytest --runslow tests/

# Run integration tests
pytest --runintegration tests/
```

### Detailed Options

```bash
# Run with coverage
pytest --cov=climate_zarr tests/

# Run with verbose output
pytest -v -s tests/

# Run specific test class
pytest tests/test_processors.py::TestBaseProcessor -v

# Run specific test method
pytest tests/test_processors.py::TestBaseProcessor::test_initialization -v
```

## Test Data

The test suite uses automatically generated synthetic data:

### Spatial Data
- **Counties**: 5 rectangular test counties in Texas
- **Grid**: 0.05° resolution covering test area
- **Projections**: WGS84 (EPSG:4326)

### Temporal Data
- **Precipitation**: 1-2 years of daily data with seasonal patterns
- **Temperature**: Realistic seasonal cycles with daily variation
- **Time Frequencies**: Daily, monthly data support

### Climate Variables
- **Precipitation**: Exponential distribution with seasonal variation
- **Temperature**: Seasonal sine waves with spatial gradients
- **Units**: Proper scientific units (kg/m²/s, K, °C)

## Expected Results

### Unit Tests
- **Coverage**: >95% code coverage for individual modules
- **Performance**: <1 second per test method
- **Reliability**: No flaky tests, deterministic results

### Integration Tests
- **Functionality**: All module interactions work correctly
- **Data Flow**: Proper data transformation through pipeline
- **Error Handling**: Graceful handling of edge cases

### End-to-End Tests
- **Workflows**: Complete processing pipelines work
- **File I/O**: Proper reading/writing of all file formats
- **Performance**: Reasonable memory usage and processing time

### Compatibility Tests
- **API Stability**: No breaking changes to public interface
- **Import Compatibility**: All expected imports work
- **Data Compatibility**: Output format unchanged

## Test Fixtures

### Shared Fixtures (conftest.py)
- `temp_test_dir`: Temporary directory for test session
- `sample_counties_gdf`: GeoDataFrame with test counties
- `sample_shapefile`: Shapefile on disk
- `sample_precipitation_xarray`: Precipitation DataArray
- `sample_temperature_xarray`: Temperature DataArray
- `sample_precipitation_zarr`: Precipitation zarr dataset
- `sample_temperature_zarr`: Temperature zarr dataset

### Specialized Fixtures
- `sample_county_info`: County metadata dictionary
- `sample_daily_precipitation`: Daily precipitation values
- `sample_daily_temperature`: Daily temperature values

## Performance Benchmarks

### Unit Tests
- Individual processor initialization: <10ms
- Strategy selection: <1ms
- Utility function calls: <1ms

### Integration Tests
- Small dataset processing: <5 seconds
- Memory usage increase: <100MB
- File I/O operations: <1 second

### End-to-End Tests
- Full workflow (5 counties, 1 year): <30 seconds
- Memory usage: <500MB peak
- Output file generation: <1 second

## Debugging Tests

### Common Issues

1. **Import Errors**: Check module paths and dependencies
2. **Fixture Errors**: Verify temporary directories and file creation
3. **Data Errors**: Check synthetic data generation
4. **Performance Issues**: Monitor memory usage and processing time

### Debugging Commands

```bash
# Run single test with full output
pytest tests/test_processors.py::TestBaseProcessor::test_initialization -v -s

# Run with debugger
pytest --pdb tests/test_processors.py::TestBaseProcessor::test_initialization

# Run with profiling
pytest --profile tests/test_processors.py

# Check test coverage
pytest --cov=climate_zarr --cov-report=html tests/
```

## Adding New Tests

### Test Structure Template

```python
import pytest
from climate_zarr.processors import NewProcessor

class TestNewProcessor:
    """Test the new processor functionality."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = NewProcessor(n_workers=2)
        assert processor.n_workers == 2
    
    def test_process_data(self, sample_data, sample_counties):
        """Test data processing."""
        processor = NewProcessor()
        results = processor.process_data(sample_data, sample_counties)
        assert len(results) > 0
    
    def test_error_handling(self):
        """Test error handling."""
        processor = NewProcessor()
        with pytest.raises(ValueError):
            processor.process_data(invalid_data)
```

### Integration Test Template

```python
def test_new_integration_workflow(self, sample_shapefile, sample_zarr):
    """Test new integration workflow."""
    with ModernCountyProcessor() as processor:
        gdf = processor.prepare_shapefile(sample_shapefile)
        results = processor.process_zarr_data(
            zarr_path=sample_zarr,
            gdf=gdf,
            scenario='test',
            variable='new_var'
        )
        assert len(results) > 0
```

## Continuous Integration

The test suite is designed to run in CI/CD environments:

### GitHub Actions Example
```yaml
name: Test Modular Architecture
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      - name: Run unit tests
        run: pytest -m unit tests/
      - name: Run integration tests
        run: pytest --runintegration tests/
      - name: Run compatibility tests
        run: pytest tests/test_backward_compatibility.py
```

## Test Maintenance

### Regular Tasks
1. **Update fixtures** when adding new climate variables
2. **Add performance benchmarks** for new processing strategies
3. **Update compatibility tests** when changing public API
4. **Review test coverage** and add tests for uncovered code

### Best Practices
1. **Deterministic tests**: Use fixed random seeds
2. **Isolated tests**: Each test should be independent
3. **Clear assertions**: Test specific behaviors, not implementation details
4. **Performance bounds**: Set reasonable time/memory limits
5. **Error coverage**: Test both success and failure cases

## Conclusion

This comprehensive testing suite ensures the modular climate-zarr architecture is:
- **Reliable**: Thoroughly tested components
- **Maintainable**: Clear test organization and documentation
- **Compatible**: Backward compatibility preserved
- **Performant**: Performance characteristics validated
- **Extensible**: Easy to add new tests for new features

The modular approach allows for targeted testing of specific components while also validating the system as a whole, providing confidence in the refactored architecture. 