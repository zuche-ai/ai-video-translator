#!/usr/bin/env python3
"""
Test script to compare original intro with extracted intro
"""

import subprocess
import os

def extract_intro_original(audio_path: str, duration: float, output_path: str):
    """Extract intro using original format (no resampling)"""
    cmd = [
        'ffmpeg', '-y', '-i', audio_path,
        '-ss', '0.0', '-t', str(duration),
        '-c', 'copy',  # Preserve original format
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def extract_intro_resampled(audio_path: str, duration: float, output_path: str):
    """Extract intro using target format (24kHz mono s16)"""
    cmd = [
        'ffmpeg', '-y', '-i', audio_path,
        '-ss', '0.0', '-t', str(duration),
        '-ar', '24000', '-ac', '1', '-sample_fmt', 's16',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def main():
    input_file = "christtestshort.wav"
    intro_duration = 30.0  # First 30 seconds
    
    print("=== Testing Intro Extraction ===")
    print(f"Input file: {input_file}")
    print(f"Intro duration: {intro_duration}s")
    
    # Extract intro in original format
    original_intro = "intro_original.wav"
    print(f"\n1. Extracting intro in original format: {original_intro}")
    extract_intro_original(input_file, intro_duration, original_intro)
    
    # Extract intro in resampled format (what we're currently doing)
    resampled_intro = "intro_resampled.wav"
    print(f"2. Extracting intro in resampled format: {resampled_intro}")
    extract_intro_resampled(input_file, intro_duration, resampled_intro)
    
    # Check file sizes and durations
    print("\n=== File Comparison ===")
    for file_path in [original_intro, resampled_intro]:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', file_path
            ], capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            print(f"{file_path}: {size:,} bytes, {duration:.2f}s")
    
    # Check audio properties
    print("\n=== Audio Properties ===")
    for file_path in [input_file, original_intro, resampled_intro]:
        if os.path.exists(file_path):
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'stream=sample_rate,channels,sample_fmt',
                '-of', 'csv=p=0', file_path
            ], capture_output=True, text=True, check=True)
            print(f"{file_path}: {result.stdout.strip()}")

if __name__ == "__main__":
    main() 