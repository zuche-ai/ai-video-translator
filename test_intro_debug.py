#!/usr/bin/env python3
"""
Detailed debug script to identify intro noise issues
"""

import subprocess
import os
import tempfile
import shutil

def extract_intro_exact(audio_path: str, duration: float, output_path: str):
    """Extract intro using exact copy method"""
    cmd = [
        'ffmpeg', '-y', '-i', audio_path,
        '-ss', '0.0', '-t', str(duration),
        '-c', 'copy',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def extract_intro_precise(audio_path: str, duration: float, output_path: str):
    """Extract intro using precise seeking"""
    cmd = [
        'ffmpeg', '-y', '-i', audio_path,
        '-ss', '0.0', '-t', str(duration),
        '-avoid_negative_ts', 'make_zero',
        '-c', 'copy',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def extract_intro_reencode(audio_path: str, duration: float, output_path: str):
    """Extract intro with re-encoding to same format"""
    cmd = [
        'ffmpeg', '-y', '-i', audio_path,
        '-ss', '0.0', '-t', str(duration),
        '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def compare_audio_files(file1: str, file2: str, label: str):
    """Compare two audio files"""
    print(f"\n=== {label} ===")
    
    # File sizes
    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)
    print(f"File sizes: {size1:,} vs {size2:,} bytes")
    
    # Durations
    result1 = subprocess.run([
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'csv=p=0', file1
    ], capture_output=True, text=True, check=True)
    result2 = subprocess.run([
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'csv=p=0', file2
    ], capture_output=True, text=True, check=True)
    
    dur1 = float(result1.stdout.strip())
    dur2 = float(result2.stdout.strip())
    print(f"Durations: {dur1:.2f}s vs {dur2:.2f}s")
    
    # Audio properties
    result1 = subprocess.run([
        'ffprobe', '-v', 'quiet', '-show_entries', 'stream=sample_rate,channels,sample_fmt',
        '-of', 'csv=p=0', file1
    ], capture_output=True, text=True, check=True)
    result2 = subprocess.run([
        'ffprobe', '-v', 'quiet', '-show_entries', 'stream=sample_rate,channels,sample_fmt',
        '-of', 'csv=p=0', file2
    ], capture_output=True, text=True, check=True)
    
    print(f"Audio properties: {result1.stdout.strip()} vs {result2.stdout.strip()}")
    
    # Check if files are identical
    if size1 == size2:
        print("Files have same size - checking if identical...")
        result = subprocess.run(['diff', file1, file2], capture_output=True)
        if result.returncode == 0:
            print("✅ Files are IDENTICAL")
        else:
            print("❌ Files are different despite same size")
    else:
        print("❌ Files have different sizes")

def main():
    input_file = "christtestshort.wav"
    intro_duration = 30.0
    
    print("=== Detailed Intro Debug ===")
    print(f"Input file: {input_file}")
    print(f"Intro duration: {intro_duration}s")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="intro_debug_")
    
    try:
        # Extract intro using different methods
        methods = [
            ("exact_copy", extract_intro_exact),
            ("precise_copy", extract_intro_precise),
            ("reencode", extract_intro_reencode)
        ]
        
        extracted_files = {}
        
        for method_name, extract_func in methods:
            output_path = os.path.join(temp_dir, f"intro_{method_name}.wav")
            print(f"\nExtracting intro using {method_name}...")
            extract_func(input_file, intro_duration, output_path)
            extracted_files[method_name] = output_path
        
        # Compare with original
        print("\n" + "="*50)
        print("COMPARING EXTRACTED FILES WITH ORIGINAL")
        print("="*50)
        
        for method_name, extracted_file in extracted_files.items():
            compare_audio_files(input_file, extracted_file, f"Original vs {method_name}")
        
        # Compare extracted files with each other
        print("\n" + "="*50)
        print("COMPARING EXTRACTED FILES WITH EACH OTHER")
        print("="*50)
        
        methods_list = list(extracted_files.keys())
        for i in range(len(methods_list)):
            for j in range(i+1, len(methods_list)):
                method1 = methods_list[i]
                method2 = methods_list[j]
                compare_audio_files(
                    extracted_files[method1], 
                    extracted_files[method2], 
                    f"{method1} vs {method2}"
                )
        
        # Test what our current method produces
        print("\n" + "="*50)
        print("TESTING OUR CURRENT METHOD")
        print("="*50)
        
        # Simulate our current extraction method
        current_intro = os.path.join(temp_dir, "current_intro.wav")
        cmd = [
            'ffmpeg', '-y', '-i', input_file,
            '-ss', '0.0', '-t', str(intro_duration),
            '-c', 'copy',
            current_intro
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        compare_audio_files(input_file, current_intro, "Original vs Current Method")
        
        print(f"\nAll test files saved in: {temp_dir}")
        print("You can listen to them to compare the audio quality.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Keep temp directory for manual inspection
        print(f"\nTemp directory preserved: {temp_dir}")

if __name__ == "__main__":
    main() 