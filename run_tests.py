#!/usr/bin/env python3
"""
Test runner for climate data processing project.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type='all', verbose=False, coverage=False):
    """Run tests based on type selection."""
    
    # Base pytest command
    cmd = ['pytest']
    
    # Add test directory based on type
    if test_type == 'all':
        cmd.append('tests/')
    elif test_type == 'unit':
        cmd.append('tests/unit/')
    elif test_type == 'integration':
        cmd.append('tests/integration/')
    elif test_type == 'performance':
        cmd.append('tests/performance/')
    elif test_type == 'fast':
        cmd.extend(['tests/', '-m', 'not slow and not performance'])
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Add options
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=src', '--cov-report=term-missing', '--cov-report=html'])
    
    # Run tests
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if coverage and result.returncode == 0:
        print("\nCoverage report generated in htmlcov/index.html")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run climate data processing tests')
    parser.add_argument(
        'type',
        nargs='?',
        default='all',
        choices=['all', 'unit', 'integration', 'performance', 'fast'],
        help='Type of tests to run (default: all)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '-c', '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    args = parser.parse_args()
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("Error: pytest not found. Install with: pip install pytest pytest-cov")
        return 1
    
    # Run tests
    return run_tests(args.type, args.verbose, args.coverage)


if __name__ == '__main__':
    sys.exit(main())