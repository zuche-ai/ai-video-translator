#!/usr/bin/env python3
"""
Example script demonstrating audio translation with VAD and voice cloning.

This script shows how to use the AudioTranslator class to:
1. Transcribe audio with Whisper
2. Use VAD to identify voice/non-voice segments
3. Translate the transcription
4. Generate voice-cloned audio for voice segments
5. Stitch all segments into a continuous audio file
"""

import os
import sys
import tempfile
import logging

# Add the project root to the path so we can import video_translator
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(f"Added {project_root} to Python path")
print(f"Current sys.path: {sys.path[:3]}...")  # Show first 3 entries

from video_translator.audio_translation.audio_translator import AudioTranslator, translate_audio_file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def progress_callback(step, percent, status_message):
    """Progress callback function for monitoring translation progress."""
    print(f"[PROGRESS] Step {step}: {percent}% - {status_message}")

def main():
    """Main function demonstrating audio translation."""
    
    # Example usage
    print("=== Audio Translation Example ===")
    
    # Check if we have the required input files
    audio_file = "input_audio.wav"  # Replace with your audio file
    reference_audio = "reference_audio.wav"  # Replace with your reference audio
    
    if not os.path.exists(audio_file):
        print(f"Error: Input audio file '{audio_file}' not found.")
        print("Please provide a valid audio file to translate.")
        return
    
    if not os.path.exists(reference_audio):
        print(f"Error: Reference audio file '{reference_audio}' not found.")
        print("Please provide a valid reference audio file for voice cloning.")
        return
    
    print(f"Input audio: {audio_file}")
    print(f"Reference audio: {reference_audio}")
    
    # Method 1: Using the AudioTranslator class directly
    print("\n--- Method 1: Using AudioTranslator class ---")
    try:
        translator = AudioTranslator()
        
        output_path = translator.translate_audio(
            audio_path=audio_file,
            src_lang="en",
            tgt_lang="es",
            reference_audio_path=reference_audio,
            progress_hook=progress_callback
        )
        
        print(f"Translation completed! Output: {output_path}")
        
    except Exception as e:
        print(f"Error with AudioTranslator class: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 2: Using the convenience function
    print("\n--- Method 2: Using convenience function ---")
    try:
        output_path = translate_audio_file(
            audio_path=audio_file,
            src_lang="en",
            tgt_lang="es",
            reference_audio_path=reference_audio,
            progress_hook=progress_callback
        )
        
        print(f"Translation completed! Output: {output_path}")
        
    except Exception as e:
        print(f"Error with convenience function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 