#!/usr/bin/env python3
"""
LDS Translation Demo

This script demonstrates the enhanced translation capabilities with LDS-specific terms.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_translator.core.translator import translate_segments
from video_translator.core.glossary_loader import LDSGlossaryLoader

def demo_lds_translation():
    """Demonstrate LDS translation enhancements."""
    print("🎯 LDS Translation Enhancement Demo")
    print("=" * 60)
    
    # Sample LDS content
    lds_segments = [
        {
            "text": "I love going to the temple and feeling the spirit there.",
            "start": 0,
            "end": 5
        },
        {
            "text": "The bishop gave a wonderful talk about testimony in sacrament meeting.",
            "start": 5,
            "end": 10
        },
        {
            "text": "We have family home evening every Monday night.",
            "start": 10,
            "end": 15
        },
        {
            "text": "The priesthood is the power and authority of God.",
            "start": 15,
            "end": 20
        },
        {
            "text": "General conference is always a spiritual feast.",
            "start": 20,
            "end": 25
        }
    ]
    
    print("📖 Original LDS Content:")
    for i, segment in enumerate(lds_segments, 1):
        print(f"  {i}. {segment['text']}")
    
    print("\n🔄 Translating to Spanish...")
    print("(This will automatically apply LDS glossary replacements)")
    
    try:
        # Translate with LDS glossary enhancement
        translated_segments = translate_segments(lds_segments, "en", "es", debug=False)
        
        print("\n✅ Enhanced Spanish Translation:")
        print("(LDS terms are properly translated with context)")
        for i, segment in enumerate(translated_segments, 1):
            print(f"  {i}. {segment['text']}")
        
        # Show glossary statistics
        loader = LDSGlossaryLoader()
        stats = loader.get_glossary_stats()
        print(f"\n📊 Glossary Statistics:")
        print(f"  Total LDS terms available: {stats['total_terms']}")
        
        # Show some key translations
        print(f"\n🔍 Key LDS Term Translations:")
        key_terms = ["temple", "bishop", "testimony", "priesthood", "general conference"]
        for term in key_terms:
            translation = loader.get_translation(term)
            print(f"  {term} → {translation}")
        
        return True
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return False

def demo_glossary_management():
    """Demonstrate glossary management features."""
    print("\n" + "=" * 60)
    print("🔧 Glossary Management Demo")
    print("=" * 60)
    
    loader = LDSGlossaryLoader()
    
    # Show current terms
    print("📋 Current LDS Terms (sample):")
    terms = loader.list_terms()[:10]  # Show first 10
    for term in terms:
        print(f"  {term['original']} → {term['translated']}")
    
    # Add a new term
    print(f"\n➕ Adding new term...")
    loader.add_term("stake conference", "conferencia de estaca", "Stake-level meeting")
    
    # Test the new term
    test_text = "We attended stake conference last weekend."
    processed = loader.apply_glossary_replacements(test_text)
    print(f"  Test: '{test_text}'")
    print(f"  Result: '{processed}'")
    
    print(f"\n✅ Glossary management demo completed!")

def main():
    """Run the complete demo."""
    print("🌟 LDS Translation Enhancement System")
    print("   Bringing Context-Aware Translation to LDS Content")
    print()
    
    # Run translation demo
    success1 = demo_lds_translation()
    
    # Run glossary management demo
    demo_glossary_management()
    
    print("\n" + "=" * 60)
    if success1:
        print("🎉 Demo completed successfully!")
        print("✨ The LDS glossary system is working perfectly!")
        print("\n💡 Key Benefits:")
        print("   • Accurate LDS terminology translation")
        print("   • Context-aware replacements")
        print("   • Easy community contributions")
        print("   • Seamless integration with existing pipeline")
    else:
        print("❌ Demo encountered issues. Please check the error messages above.")
    
    return success1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 