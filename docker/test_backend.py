#!/usr/bin/env python3
"""
Backend container test script
Tests core functionality of the video translator backend
"""

import sys
import requests

def test_imports():
    """Test that all core dependencies can be imported"""
    print("Testing core dependencies...")
    
    try:
        import whisper
        print("✅ Whisper imported successfully")
    except ImportError as e:
        print(f"❌ Whisper import failed: {e}")
        return False
    
    try:
        import argostranslate
        print("✅ ArgosTranslate imported successfully")
    except ImportError as e:
        print(f"❌ ArgosTranslate import failed: {e}")
        return False
    
    try:
        import TTS
        print("✅ TTS imported successfully")
    except ImportError as e:
        print(f"❌ TTS import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    print("✅ All core dependencies imported successfully")
    return True

def test_whisper_model():
    """Test that Whisper model can be loaded"""
    print("\nTesting Whisper model loading...")
    
    try:
        import whisper
        model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Whisper model loading failed: {e}")
        return False

def test_api_health():
    """Test API health endpoint"""
    print("\nTesting API health endpoint...")
    
    try:
        response = requests.get("http://localhost:5001/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ API health check passed: {response.status_code}")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Backend Container Tests")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_whisper_model,
        test_api_health
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'=' * 30}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All backend tests passed!")
        return 0
    else:
        print("❌ Some backend tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 