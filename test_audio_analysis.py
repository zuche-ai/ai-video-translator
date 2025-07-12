#!/usr/bin/env python3
"""
Audio analysis script to identify noise sources
"""

import subprocess
import os

def analyze_audio_levels(audio_path: str, label: str):
    """Analyze audio levels and properties"""
    print(f"\n=== {label} ===")
    print(f"File: {audio_path}")
    
    if not os.path.exists(audio_path):
        print("❌ File not found")
        return
    
    # File size
    size = os.path.getsize(audio_path)
    print(f"Size: {size:,} bytes")
    
    # Duration
    result = subprocess.run([
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'csv=p=0', audio_path
    ], capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())
    print(f"Duration: {duration:.2f}s")
    
    # Audio properties
    result = subprocess.run([
        'ffprobe', '-v', 'quiet', '-show_entries', 'stream=sample_rate,channels,sample_fmt',
        '-of', 'csv=p=0', audio_path
    ], capture_output=True, text=True, check=True)
    print(f"Audio: {result.stdout.strip()}")
    
    # Audio levels (RMS)
    result = subprocess.run([
        'ffmpeg', '-i', audio_path, '-af', 'volumedetect', '-f', 'null', '-'
    ], capture_output=True, text=True)
    
    # Extract mean_volume and max_volume
    output = result.stderr
    mean_volume = None
    max_volume = None
    
    for line in output.split('\n'):
        if 'mean_volume' in line:
            mean_volume = line.split(':')[1].strip()
        elif 'max_volume' in line:
            max_volume = line.split(':')[1].strip()
    
    print(f"Mean volume: {mean_volume}")
    print(f"Max volume: {max_volume}")
    
    # Check for silence
    result = subprocess.run([
        'ffmpeg', '-i', audio_path, '-af', 'silencedetect=noise=-50dB:d=0.1', '-f', 'null', '-'
    ], capture_output=True, text=True)
    
    silence_info = result.stderr
    print("Silence detection:")
    for line in silence_info.split('\n'):
        if 'silence_start' in line or 'silence_end' in line:
            print(f"  {line.strip()}")

def main():
    print("=== Audio Analysis ===")
    
    # Analyze original file
    analyze_audio_levels("christtestshort.wav", "Original File")
    
    # Analyze our output
    analyze_audio_levels("output_v3_1_fixed.wav", "Our Output")
    
    # Analyze just the intro from original
    print("\n=== Extracting and analyzing intro from original ===")
    intro_original = "intro_from_original.wav"
    cmd = [
        'ffmpeg', '-y', '-i', 'christtestshort.wav',
        '-ss', '0.0', '-t', '30.0',
        '-c', 'copy',
        intro_original
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    analyze_audio_levels(intro_original, "Intro from Original")
    
    # Analyze just the intro from our output
    print("\n=== Extracting and analyzing intro from our output ===")
    intro_output = "intro_from_output.wav"
    cmd = [
        'ffmpeg', '-y', '-i', 'output_v3_1_fixed.wav',
        '-ss', '0.0', '-t', '30.0',
        '-c', 'copy',
        intro_output
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    analyze_audio_levels(intro_output, "Intro from Our Output")
    
    # Compare the two intros
    print("\n=== Comparing Intros ===")
    if os.path.exists(intro_original) and os.path.exists(intro_output):
        result = subprocess.run(['diff', intro_original, intro_output], capture_output=True)
        if result.returncode == 0:
            print("✅ Intros are IDENTICAL")
        else:
            print("❌ Intros are different")
            print("This suggests the issue is in our processing pipeline")

if __name__ == "__main__":
    main() 