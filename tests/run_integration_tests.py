#!/usr/bin/env python
"""
Runner script for Climate Zarr integration tests.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run the integration test suite with various options."""
    
    # Base pytest command
    cmd = ["pytest", "-v"]
    
    # Add command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Run only quick tests (exclude slow and e2e)
            cmd.extend(["-m", "not slow and not e2e"])
        elif sys.argv[1] == "--cli":
            # Run only CLI tests
            cmd.extend(["-m", "cli"])
        elif sys.argv[1] == "--coverage":
            # Run with coverage
            cmd.extend(["--cov=climate_zarr", "--cov-report=term-missing", "--cov-report=html"])
        elif sys.argv[1] == "--parallel":
            # Run tests in parallel (requires pytest-xdist)
            cmd.extend(["-n", "auto"])
    
    # Run the tests
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode


if __name__ == "__main__":
    print("🧪 Climate Zarr Integration Test Suite")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python run_integration_tests.py [option]")
        print("\nOptions:")
        print("  --quick     Run only quick tests (skip slow/e2e)")
        print("  --cli       Run only CLI tests")
        print("  --coverage  Run with coverage report")
        print("  --parallel  Run tests in parallel")
        print("  --help      Show this help message")
        sys.exit(0)
    
    exit_code = run_tests()
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(exit_code)