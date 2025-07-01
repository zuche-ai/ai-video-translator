#!/usr/bin/env python3
"""
Test runner for video translator project.
Runs all unit tests and provides a summary.
"""

import unittest
import sys
import os

def run_all_tests():
    """Run all unit tests and return results."""
    # Get the directory containing this script
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def main():
    """Main function to run tests and display results."""
    print("🧪 Running Video Translator Unit Tests")
    print("=" * 50)
    
    result = run_all_tests()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} test(s) failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 