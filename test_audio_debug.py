#!/usr/bin/env python3
"""
Simple test script to debug audio translation pipeline.
Run this from the project root directory.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def progress_callback(step, percent, status_message):
    """Progress callback function for monitoring translation progress."""
    print(f"[PROGRESS] Step {step}: {percent}% - {status_message}")

def main():
    """Main function to test audio translation."""
    
    print("=== Audio Translation Debug Test ===")
    
    # Check if we have the required input files
    audio_file = "christtestshort.wav"  # Use your existing file
    reference_audio = "christtestshort.wav"  # Use same file as reference
    
    if not os.path.exists(audio_file):
        print(f"Error: Input audio file '{audio_file}' not found.")
        print("Please make sure christtestshort.wav is in the current directory.")
        return
    
    print(f"Input audio: {audio_file}")
    print(f"Reference audio: {reference_audio}")
    
    try:
        # Import the module
        print("Importing video_translator modules...")
        from video_translator.audio_translation.audio_translator import AudioTranslator, translate_audio_file
        print("Import successful!")
        
        # Test the translation
        print("\n--- Testing Audio Translation ---")
        output_path = translate_audio_file(
            audio_path=audio_file,
            src_lang="en",
            tgt_lang="es",
            reference_audio_path=reference_audio,
            progress_hook=progress_callback
        )
        
        print(f"Translation completed! Output: {output_path}")
        
        # Check if output file exists and has content
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"Output file size: {size} bytes")
        else:
            print("Warning: Output file not found!")
            
    except Exception as e:
        print(f"Error during translation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 