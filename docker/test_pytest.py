#!/usr/bin/env python3
"""
Pytest runner for container tests
"""

import subprocess
import sys

def main():
    try:
        import pytest
        print("✅ Pytest is available")
        
        # Run pytest
        result = subprocess.run([
            "python", "-m", "pytest", "tests/", "-v", "--tb=short", "--timeout=30"
        ], capture_output=True, text=True, cwd="/app")
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✅ All pytest tests passed!")
            return 0
        else:
            print(f"❌ Some pytest tests failed (exit code: {result.returncode})")
            return 1
            
    except ImportError:
        print("ℹ️  Pytest not available, skipping unit tests")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 