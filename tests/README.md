# Climate Zarr Integration Test Suite

This directory contains integration tests for the Climate Zarr toolkit. These tests verify that all components work correctly together in real-world scenarios.

## What are Integration Tests?

Integration tests verify that different parts of the application work correctly together, testing:
- Complete workflows from input to output
- Component interactions and data flow
- Real file I/O operations
- Error handling across module boundaries
- Performance characteristics

## Test Structure

```
tests/
├── conftest.py                      # Pytest fixtures and configuration
├── test_cli_integration.py          # CLI command tests
├── test_zarr_conversion_integration.py  # NetCDF to Zarr conversion tests
├── test_county_stats_integration.py # County statistics calculation tests
├── test_end_to_end_integration.py   # Complete workflow tests
└── run_integration_tests.py         # Test runner script
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_cli_integration.py

# Run specific test
pytest tests/test_cli_integration.py::TestCLIIntegration::test_create_zarr_no_interactive
```

### Using the Test Runner

```bash
# Run all tests
python tests/run_integration_tests.py

# Run quick tests only (skip slow/e2e)
python tests/run_integration_tests.py --quick

# Run with coverage report
python tests/run_integration_tests.py --coverage

# Run tests in parallel
python tests/run_integration_tests.py --parallel
```

### Test Markers

Tests are marked with categories for selective execution:

```bash
# Run only CLI tests
pytest -m cli

# Run only conversion tests
pytest -m conversion

# Skip slow tests
pytest -m "not slow"

# Skip end-to-end tests
pytest -m "not e2e"
```

## Test Fixtures

The test suite provides several fixtures for common test data:

- `sample_netcdf_files`: Creates synthetic NetCDF climate data files
- `sample_shapefile`: Creates mock county boundary shapefiles
- `zarr_output_dir`: Temporary directory for Zarr output
- `stats_output_dir`: Temporary directory for statistics output
- `cli_runner`: Typer CLI test runner
- `mock_climate_config`: Sample configuration object

## Coverage

To generate a coverage report:

```bash
# Terminal report
pytest --cov=climate_zarr --cov-report=term-missing

# HTML report
pytest --cov=climate_zarr --cov-report=html
# Open htmlcov/index.html in a browser
```

## Writing New Tests

When adding new integration tests:

1. Use appropriate fixtures for test data
2. Test complete workflows, not isolated functions
3. Verify file outputs and data integrity
4. Add appropriate markers for test categorization
5. Clean up temporary files (fixtures handle this automatically)

Example:

```python
@pytest.mark.integration
@pytest.mark.conversion
def test_new_feature(sample_netcdf_files, zarr_output_dir):
    """Test description."""
    # Arrange
    output_path = zarr_output_dir / "test_output.zarr"
    
    # Act
    result = process_data(sample_netcdf_files, output_path)
    
    # Assert
    assert output_path.exists()
    assert result.success
```

## Performance Considerations

Integration tests use small synthetic datasets to run quickly while still testing real workflows. The fixtures create:
- 3 years of daily climate data
- 20x25 spatial grid
- 3 mock counties

This keeps test execution time reasonable while exercising all code paths.