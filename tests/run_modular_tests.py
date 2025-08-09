#!/usr/bin/env python
"""Test runner for modular climate-zarr tests."""

import sys
import subprocess
from pathlib import Path


def run_test_suite():
    """Run the complete test suite for modular climate-zarr."""
    
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    # Define test categories and their files
    test_categories = {
        "Unit Tests": [
            "test_processors.py",
            "test_strategies.py", 
            "test_utils.py"
        ],
        "Integration Tests": [
            "test_modular_integration.py",
            "test_end_to_end_modular.py"
        ],
        "Compatibility Tests": [
            "test_backward_compatibility.py"
        ]
    }
    
    # Run each category
    all_passed = True
    
    for category, test_files in test_categories.items():
        print(f"\n{'='*60}")
        print(f"Running {category}")
        print(f"{'='*60}")
        
        for test_file in test_files:
            test_path = tests_dir / test_file
            
            if not test_path.exists():
                print(f"⚠️  Test file not found: {test_file}")
                continue
            
            print(f"\n🧪 Running {test_file}...")
            
            # Run pytest on the specific file
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(test_path), 
                "-v", 
                "--tb=short",
                "--color=yes"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test_file} passed")
            else:
                print(f"❌ {test_file} failed")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


def run_specific_test(test_name):
    """Run a specific test file."""
    tests_dir = Path(__file__).parent
    test_path = tests_dir / test_name
    
    if not test_path.exists():
        print(f"❌ Test file not found: {test_name}")
        return 1
    
    print(f"🧪 Running {test_name}...")
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        str(test_path), 
        "-v", 
        "--tb=long",
        "--color=yes"
    ])
    
    return result.returncode


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        if not test_name.endswith('.py'):
            test_name += '.py'
        return run_specific_test(test_name)
    else:
        # Run full test suite
        return run_test_suite()


if __name__ == "__main__":
    sys.exit(main()) 