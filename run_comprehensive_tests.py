#!/usr/bin/env python3
"""
Comprehensive test runner for the video translation API.
This script runs all unit tests and provides detailed reporting.
"""

import unittest
import sys
import os
import time
import traceback
from io import StringIO

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_test_suite():
    """Run all test suites and provide detailed reporting"""
    
    # Discover and load all test modules
    test_modules = [
        'tests.test_api_comprehensive',
        'tests.test_api_integration', 
        'tests.test_audio_processing'
    ]
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    print("🔍 Loading test modules...")
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=['*'])
            module_suite = loader.loadTestsFromModule(module)
            suite.addTests(module_suite)
            print(f"✅ Loaded {module_name}: {module_suite.countTestCases()} tests")
        except ImportError as e:
            print(f"❌ Failed to load {module_name}: {e}")
        except Exception as e:
            print(f"❌ Error loading {module_name}: {e}")
    
    print(f"\n📊 Total tests to run: {suite.countTestCases()}")
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("\n🚀 Starting test execution...")
    print("=" * 80)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print detailed results
    print("\n" + "=" * 80)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    # Calculate success rate
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    # Print detailed failure information
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        print("-" * 40)
        for test, traceback_str in result.failures:
            print(f"Test: {test}")
            print(f"Error: {traceback_str}")
            print()
    
    if result.errors:
        print(f"\n⚠️  ERRORS ({len(result.errors)}):")
        print("-" * 40)
        for test, traceback_str in result.errors:
            print(f"Test: {test}")
            print(f"Error: {traceback_str}")
            print()
    
    # Print test categories
    print("\n📊 TEST CATEGORIES:")
    print("-" * 40)
    
    category_stats = {
        'API Functions': 0,
        'Audio Processing': 0,
        'TTS Generation': 0,
        'Voice Cloning': 0,
        'SRT Parsing': 0,
        'Integration': 0,
        'Performance': 0,
        'Edge Cases': 0
    }
    
    # Count tests by category (based on test method names)
    for test in suite:
        test_name = test._testMethodName.lower()
        if 'api' in test_name or 'endpoint' in test_name:
            category_stats['API Functions'] += 1
        elif 'audio' in test_name or 'splice' in test_name:
            category_stats['Audio Processing'] += 1
        elif 'tts' in test_name:
            category_stats['TTS Generation'] += 1
        elif 'voice' in test_name or 'clone' in test_name:
            category_stats['Voice Cloning'] += 1
        elif 'srt' in test_name or 'parse' in test_name:
            category_stats['SRT Parsing'] += 1
        elif 'integration' in test_name:
            category_stats['Integration'] += 1
        elif 'performance' in test_name or 'memory' in test_name:
            category_stats['Performance'] += 1
        elif 'edge' in test_name:
            category_stats['Edge Cases'] += 1
    
    for category, count in category_stats.items():
        if count > 0:
            print(f"{category}: {count} tests")
    
    # Recommendations based on results
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("🎉 All tests passed! The codebase appears to be in good shape.")
        print("   Consider adding more edge case tests for robustness.")
    else:
        print("🔧 Issues detected. Recommendations:")
        
        if len(result.failures) > 0:
            print("   - Review and fix failing tests")
            print("   - Check for logic errors in the implementation")
        
        if len(result.errors) > 0:
            print("   - Fix import and dependency issues")
            print("   - Ensure all required modules are available")
        
        if success_rate < 80:
            print("   - Low success rate indicates significant issues")
            print("   - Consider running tests in isolation to identify problems")
    
    # Performance recommendations
    if end_time - start_time > 60:
        print("   - Test execution is slow, consider optimizing test setup")
    
    return result.wasSuccessful()


def run_specific_test_category(category):
    """Run tests for a specific category"""
    categories = {
        'api': 'tests.test_api_comprehensive',
        'integration': 'tests.test_api_integration',
        'audio': 'tests.test_audio_processing',
        'all': None
    }
    
    if category not in categories:
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return False
    
    if category == 'all':
        return run_test_suite()
    
    # Run specific category
    module_name = categories[category]
    try:
        module = __import__(module_name, fromlist=['*'])
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        print(f"🔍 Running {category} tests...")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except Exception as e:
        print(f"❌ Error running {category} tests: {e}")
        return False


def main():
    """Main function"""
    print("🧪 VIDEO TRANSLATION API - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        category = sys.argv[1].lower()
        success = run_specific_test_category(category)
    else:
        success = run_test_suite()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    main() 