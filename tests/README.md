# Climate Data Processing Tests

## Test Organization

The test suite is organized into three main categories:

### 1. Unit Tests (`unit/`)
- **test_climate_calculator.py**: Tests core climate calculation logic
- Fast, isolated tests of individual components
- No external dependencies required

### 2. Integration Tests (`integration/`)
- **test_parallel_processing.py**: Tests parallel processing functionality
- **test_fixed_baseline.py**: Tests fixed baseline percentile calculations
- **test_modular_processor.py**: Tests modular design components
- Require access to climate data files

### 3. Performance Tests (`performance/`)
- **test_optimization.py**: Tests optimization effectiveness
- **test_performance_optimization.py**: Compares original vs optimized performance
- Measure speedups and resource usage

### Archive (`archive/`)
Contains older test files kept for reference but not actively maintained.

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-cov
```

### Running All Tests
```bash
# From project root
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Running Specific Test Categories
```bash
# Only unit tests (fast)
pytest tests/unit/

# Only integration tests
pytest tests/integration/

# Only performance tests
pytest tests/performance/

# Exclude slow tests
pytest tests/ -m "not slow"
```

### Running Individual Test Files
```bash
# Run specific test file
pytest tests/unit/test_climate_calculator.py

# Run specific test
pytest tests/unit/test_climate_calculator.py::TestClimateIndicatorCalculator::test_calculate_indicators
```

## Test Configuration

Tests use `conftest.py` for shared fixtures and configuration:

- **base_data_path**: Path to climate data (from environment or default)
- **shapefile_path**: Path to county shapefile
- **test_counties**: Standard sets of test counties (small/medium/large)
- **test_periods**: Standard time periods for testing
- **climate_variables**: Standard climate variables
- **climate_scenarios**: Standard scenarios

## Environment Variables

Set these before running tests:

```bash
# Custom climate data path
export CLIMATE_DATA_PATH=/path/to/your/climate/data

# Run tests
pytest tests/
```

## Writing New Tests

### 1. Choose the appropriate directory:
- `unit/` for testing individual functions/methods
- `integration/` for testing component interactions
- `performance/` for benchmarking and optimization tests

### 2. Use appropriate markers:
```python
@pytest.mark.unit         # For unit tests
@pytest.mark.integration  # For integration tests
@pytest.mark.performance  # For performance tests
@pytest.mark.slow        # For slow-running tests
```

### 3. Use fixtures for common data:
```python
def test_my_feature(shapefile_path, test_counties, test_periods):
    # Use standard test data
    counties = test_counties['small']
    period = test_periods['quick']
```

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Fast tests only (no slow/performance tests)
pytest tests/ -m "not slow and not performance"

# With XML output for CI
pytest tests/ --junitxml=test-results.xml
```

## Troubleshooting

### Tests fail due to missing data
- Ensure climate data path is correct
- Check that shapefile exists in `data/shapefiles/`
- Verify file permissions

### Import errors
- Run tests from project root
- Ensure `src` is in Python path (handled by conftest.py)

### Performance tests take too long
- Use smaller test datasets
- Skip with `pytest -m "not slow"`
- Reduce number of test counties/years