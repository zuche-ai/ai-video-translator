#!/usr/bin/env python3
"""
Test script for the new VAD-first audio translation approach.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from video_translator.audio_translation.audio_translator import AudioTranslator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def progress_hook(current_step: int, total_steps: int, message: str):
    """Progress callback for the translation process."""
    print(f"Progress: {current_step}/{total_steps} - {message}")

def main():
    """Test the VAD-first audio translation approach."""
    
    # Test audio file
    audio_file = "christest.wav"
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        print("Please make sure the test audio file exists in the current directory.")
        return
    
    # Reference audio for voice cloning (use the same file for testing)
    reference_audio = audio_file
    
    print(f"Testing VAD-first audio translation with: {audio_file}")
    print(f"Reference audio: {reference_audio}")
    print("-" * 50)
    
    try:
        # Initialize the audio translator
        translator = AudioTranslator()
        
        # Test the new VAD-first approach
        output_path = translator.translate_audio_vad_first(
            audio_path=audio_file,
            src_lang="en",
            tgt_lang="es",
            reference_audio_path=reference_audio,
            progress_hook=progress_hook
        )
        
        print(f"\n✅ VAD-first translation completed successfully!")
        print(f"Output file: {output_path}")
        
        # Check if output file exists and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"Output file size: {file_size} bytes")
            
            if file_size > 0:
                print("✅ Output file has content!")
            else:
                print("⚠️  Output file is empty!")
        else:
            print("❌ Output file was not created!")
            
    except Exception as e:
        print(f"❌ VAD-first translation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 