#!/usr/bin/env python3
"""
Simple test script to debug audio translation pipeline using the new simple approach.
Run this from the project root directory.
"""

import os
import sys
import logging
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def progress_callback(step, percent, status_message):
    """Progress callback function for monitoring translation progress."""
    print(f"[PROGRESS] Step {step}: {percent}% - {status_message}")

def test_simple_approach():
    """Test the new simple voice cloning approach."""
    
    print("=== Testing Simple Voice Cloning Approach ===")
    
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
        from video_translator.audio_translation.audio_translator import AudioTranslator
        from video_translator.core.transcriber import transcribe_video
        from video_translator.core.translator import translate_segments
        print("Import successful!")
        
        # Create translator instance
        translator = AudioTranslator()
        
        # Step 1: Transcribe
        print("\n--- Step 1: Transcribing Audio ---")
        segments = transcribe_video(audio_file, language="en", debug=True)
        print(f"Transcription completed with {len(segments)} segments")
        
        # Step 2: Translate
        print("\n--- Step 2: Translating Text ---")
        translated_segments = translate_segments(segments, "en", "es", debug=True)
        print(f"Translation completed for {len(translated_segments)} segments")
        
        # Step 3: Test the simple voice cloning approach
        print("\n--- Step 3: Testing Simple Voice Cloning ---")
        
        # Create temporary directory for intermediate files
        import tempfile
        temp_dir = tempfile.mkdtemp()
        voice_segments_dir = os.path.join(temp_dir, "voice_segments")
        non_voice_segments_dir = os.path.join(temp_dir, "non_voice_segments")
        
        for dir_path in [voice_segments_dir, non_voice_segments_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Use the new simple approach
        audio_files = translator._generate_voice_audio_simple(
            segments=translated_segments,
            reference_audio_path=reference_audio,
            output_dir=voice_segments_dir,
            language="es"
        )
        
        print(f"Simple approach completed! Generated {len(audio_files)} audio files:")
        for i, audio_file_path in enumerate(audio_files):
            if os.path.exists(audio_file_path):
                size = os.path.getsize(audio_file_path)
                print(f"  File {i+1}: {os.path.basename(audio_file_path)} ({size} bytes)")
                
                # Check audio properties
                try:
                    import librosa
                    audio, sr = librosa.load(audio_file_path, sr=None)
                    duration = len(audio) / sr
                    print(f"    Duration: {duration:.2f}s, Sample rate: {sr}Hz")
                    
                    # Check for silence at start/end
                    silence_threshold = 0.01
                    start_silence = 0
                    end_silence = 0
                    
                    # Check start silence
                    for j in range(min(1000, len(audio))):
                        if abs(audio[j]) > silence_threshold:
                            start_silence = j / sr
                            break
                    
                    # Check end silence
                    for j in range(len(audio) - 1, max(0, len(audio) - 1000), -1):
                        if abs(audio[j]) > silence_threshold:
                            end_silence = (len(audio) - j) / sr
                            break
                    
                    if start_silence > 0.1:
                        print(f"    WARNING: {start_silence:.2f}s silence at start")
                    if end_silence > 0.1:
                        print(f"    WARNING: {end_silence:.2f}s silence at end")
                        
                except Exception as e:
                    print(f"    Could not analyze audio: {e}")
            else:
                print(f"  File {i+1}: {os.path.basename(audio_file_path)} (NOT FOUND)")
        
        # Step 3.5: Clean audio files (remove silence, normalize)
        print("\n--- Step 3.5: Cleaning Audio Files ---")
        cleaned_audio_files = []
        
        for i, audio_file_path in enumerate(audio_files):
            if os.path.exists(audio_file_path):
                cleaned_path = audio_file_path.replace('.wav', '_cleaned.wav')
                
                # Use ffmpeg to remove silence and normalize
                cmd = [
                    'ffmpeg', '-y',
                    '-i', audio_file_path,
                    '-af', 'silenceremove=1:0:-50dB,areverse,silenceremove=1:0:-50dB,areverse,loudnorm=I=-16:TP=-1.5:LRA=11',  # Remove silence and normalize
                    '-ar', '22050',  # Set sample rate
                    cleaned_path
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    cleaned_audio_files.append(cleaned_path)
                    print(f"  Cleaned file {i+1}: {os.path.basename(cleaned_path)}")
                except subprocess.CalledProcessError as e:
                    print(f"  Failed to clean file {i+1}: {e}")
                    cleaned_audio_files.append(audio_file_path)  # Use original if cleaning fails
            else:
                cleaned_audio_files.append(audio_file_path)
        
        # Step 4: Simple audio stitching
        print("\n--- Step 4: Simple Audio Stitching ---")
        
        # Save final audio to current directory so it doesn't get deleted
        output_filename = "test_output_simple_approach.wav"
        output_path = os.path.join(os.getcwd(), output_filename)
        
        final_audio_path = translator._stitch_audio_simple(
            voice_files=cleaned_audio_files,  # Use cleaned files
            output_path=output_path
        )
        
        print(f"Final audio stitched! Output: {final_audio_path}")
        
        # Check final output
        if os.path.exists(final_audio_path):
            size = os.path.getsize(final_audio_path)
            print(f"Final audio file size: {size} bytes")
            print(f"Final audio saved as: {final_audio_path}")
            print(f"Final audio saved in current directory: {os.getcwd()}")
        else:
            print("Warning: Final audio file not found!")
        
        print(f"\nTemporary files saved in: {temp_dir}")
        print("Note: Temporary files will remain until manually deleted.")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to test audio translation."""
    
    print("=== Audio Translation Debug Test (Simple Approach) ===")
    
    # Test the simple approach
    test_simple_approach()
    
    print("\n=== Test Complete ===")
    print("The simple approach should have generated audio files without character limit warnings.")

if __name__ == "__main__":
    main() 