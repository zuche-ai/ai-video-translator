#!/usr/bin/env python3
"""
Comparison test script for old vs new VAD-first audio translation approaches.
"""

import os
import sys
import logging
import time
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

def test_old_approach(translator, audio_file, reference_audio):
    """Test the old approach (Whisper first, then VAD)."""
    print("\n🔄 Testing OLD approach (Whisper first, then VAD)...")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        output_path = translator.translate_audio(
            audio_path=audio_file,
            src_lang="en",
            tgt_lang="es",
            reference_audio_path=reference_audio,
            output_path="old_approach_output.wav",
            progress_hook=progress_hook
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Old approach completed in {duration:.2f} seconds")
        print(f"Output file: {output_path}")
        
        # Check output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"Output file size: {file_size} bytes")
            return True, duration, file_size
        else:
            print("❌ Output file was not created!")
            return False, duration, 0
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Old approach failed: {e}")
        return False, duration, 0

def test_new_approach(translator, audio_file, reference_audio):
    """Test the new VAD-first approach."""
    print("\n🔄 Testing NEW approach (VAD first, then Whisper)...")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        output_path = translator.translate_audio_vad_first(
            audio_path=audio_file,
            src_lang="en",
            tgt_lang="es",
            reference_audio_path=reference_audio,
            output_path="new_approach_output.wav",
            progress_hook=progress_hook
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ New approach completed in {duration:.2f} seconds")
        print(f"Output file: {output_path}")
        
        # Check output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"Output file size: {file_size} bytes")
            return True, duration, file_size
        else:
            print("❌ Output file was not created!")
            return False, duration, 0
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ New approach failed: {e}")
        return False, duration, 0

def main():
    """Compare the old and new approaches."""
    
    # Test audio file
    audio_file = "christest.wav"
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        print("Please make sure the test audio file exists in the current directory.")
        return
    
    # Reference audio for voice cloning (use the same file for testing)
    reference_audio = audio_file
    
    print(f"Comparing audio translation approaches with: {audio_file}")
    print(f"Reference audio: {reference_audio}")
    print("=" * 60)
    
    # Initialize the audio translator
    translator = AudioTranslator()
    
    # Test old approach
    old_success, old_duration, old_size = test_old_approach(translator, audio_file, reference_audio)
    
    # Test new approach
    new_success, new_duration, new_size = test_new_approach(translator, audio_file, reference_audio)
    
    # Print comparison results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"Old approach (Whisper first):")
    print(f"  Success: {'✅' if old_success else '❌'}")
    print(f"  Duration: {old_duration:.2f} seconds")
    print(f"  Output size: {old_size} bytes")
    
    print(f"\nNew approach (VAD first):")
    print(f"  Success: {'✅' if new_success else '❌'}")
    print(f"  Duration: {new_duration:.2f} seconds")
    print(f"  Output size: {new_size} bytes")
    
    if old_success and new_success:
        print(f"\nSpeed comparison:")
        if new_duration < old_duration:
            speedup = old_duration / new_duration
            print(f"  New approach is {speedup:.2f}x faster!")
        else:
            slowdown = new_duration / old_duration
            print(f"  New approach is {slowdown:.2f}x slower")
        
        print(f"\nSize comparison:")
        if new_size > 0 and old_size > 0:
            size_ratio = new_size / old_size
            print(f"  New output is {size_ratio:.2f}x the size of old output")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 