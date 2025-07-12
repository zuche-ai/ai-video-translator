#!/usr/bin/env python3
"""
Test script for Multi-Speaker Translator V3.1 with fixed voice/music separation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_translator.audio_translation.multi_speaker_translator_v3_1 import MultiSpeakerTranslatorV3_1

def main():
    input_file = "christtestshort.wav"
    output_file = "output_v3_1_fixed.wav"
    target_language = "es"
    
    print(f"Starting multi-speaker translation...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Target language: {target_language}")
    
    try:
        # Create translator instance
        translator = MultiSpeakerTranslatorV3_1()
        
        # Test non-voice detection separately
        print("\n=== Testing Non-Voice Detection ===")
        non_voice_segments = translator._detect_non_voice_segments(input_file)
        print(f"Detected {len(non_voice_segments)} non-voice segments:")
        for i, seg in enumerate(non_voice_segments):
            print(f"  {i+1}: {seg['type']} - {seg['start']:.2f}s to {seg['end']:.2f}s ({seg['duration']:.2f}s)")
        
        # Run full translation
        print("\n=== Running Full Translation ===")
        result = translator.translate_audio(
            audio_path=input_file,
            src_lang="en",
            tgt_lang=target_language,
            output_path=output_file
        )
        
        print(f"Translation completed successfully!")
        print(f"Output file: {result}")
        
        # Check if file exists and get size
        if os.path.exists(result):
            size = os.path.getsize(result)
            print(f"File size: {size:,} bytes")
            
            # Get duration
            import subprocess
            duration_result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', result
            ], capture_output=True, text=True, check=True)
            duration = float(duration_result.stdout.strip())
            print(f"Duration: {duration:.2f} seconds")
        else:
            print("Warning: Output file not found!")
            
    except Exception as e:
        print(f"Error during translation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 