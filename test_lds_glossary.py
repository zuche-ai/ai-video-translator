#!/usr/bin/env python3
"""
Test script for LDS Glossary functionality

This script tests the LDS glossary loader and its integration with the translator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_translator.core.glossary_loader import LDSGlossaryLoader, apply_lds_glossary
from video_translator.core.translator import translate_segments

def test_glossary_loader():
    """Test the glossary loader functionality."""
    print("=== Testing LDS Glossary Loader ===")
    
    # Test loading the glossary
    loader = LDSGlossaryLoader()
    
    # Get statistics
    stats = loader.get_glossary_stats()
    print(f"Loaded {stats['total_terms']} terms")
    
    # Test some specific translations
    test_terms = [
        "temple",
        "ward",
        "bishop",
        "testimony",
        "priesthood",
        "general conference",
        "family home evening"
    ]
    
    print("\nTesting individual term translations:")
    for term in test_terms:
        translation = loader.get_translation(term)
        print(f"  {term} -> {translation}")
    
    # Test text replacement
    test_text = "I love going to the temple and attending sacrament meeting. The bishop gave a great talk about testimony."
    print(f"\nOriginal text: {test_text}")
    
    processed_text = loader.apply_glossary_replacements(test_text)
    print(f"Processed text: {processed_text}")
    
    return True

def test_translator_integration():
    """Test the translator integration with LDS glossary."""
    print("\n=== Testing Translator Integration ===")
    
    # Test segments with LDS terms
    test_segments = [
        {"text": "I love going to the temple.", "start": 0, "end": 5},
        {"text": "The bishop gave a great talk about testimony.", "start": 5, "end": 10},
        {"text": "We have sacrament meeting every Sunday.", "start": 10, "end": 15}
    ]
    
    print("Original segments:")
    for i, segment in enumerate(test_segments):
        print(f"  {i+1}: {segment['text']}")
    
    try:
        # Translate to Spanish (this will trigger LDS glossary replacement)
        translated_segments = translate_segments(test_segments, "en", "es", debug=True)
        
        print("\nTranslated segments (with LDS glossary applied):")
        for i, segment in enumerate(translated_segments):
            print(f"  {i+1}: {segment['text']}")
        
        return True
        
    except Exception as e:
        print(f"Translation test failed: {e}")
        return False

def test_glossary_management():
    """Test glossary management functions."""
    print("\n=== Testing Glossary Management ===")
    
    loader = LDSGlossaryLoader()
    
    # Test adding a new term
    print("Adding new term: 'stake conference' -> 'conferencia de estaca'")
    loader.add_term("stake conference", "conferencia de estaca", "Stake-level meeting")
    
    # Test the new term
    translation = loader.get_translation("stake conference")
    print(f"Translation: stake conference -> {translation}")
    
    # Test in context
    test_text = "We attended stake conference last weekend."
    processed_text = loader.apply_glossary_replacements(test_text)
    print(f"Context test: '{test_text}' -> '{processed_text}'")
    
    return True

def main():
    """Run all tests."""
    print("LDS Glossary Test Suite")
    print("=" * 50)
    
    tests = [
        ("Glossary Loader", test_glossary_loader),
        ("Translator Integration", test_translator_integration),
        ("Glossary Management", test_glossary_management)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ {test_name}: FAILED - {e}")
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The LDS glossary is working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 