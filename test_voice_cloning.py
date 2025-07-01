#!/usr/bin/env python3
"""
Test script to verify voice cloning with mounted models.
"""

import requests
import json
import time
import os

def test_voice_cloning():
    """Test voice cloning with mounted models."""
    
    # Create a simple test audio file (we'll use a placeholder for now)
    test_audio_path = "test_audio.wav"
    
    # Create a minimal test audio file using ffmpeg
    os.system(f"ffmpeg -f lavfi -i 'sine=frequency=440:duration=2' -ar 22050 {test_audio_path} -y")
    
    if not os.path.exists(test_audio_path):
        print("Failed to create test audio file")
        return
    
    print(f"Created test audio file: {test_audio_path}")
    
    # Test the voice cloning endpoint
    url = "http://localhost:5001/clone_voice"
    
    with open(test_audio_path, 'rb') as f:
        files = {'audio': f}
        data = {
            'text': 'Esta es una prueba de voz en español.',
            'language': 'es',
            'speed': '1.0'
        }
        
        print("Sending voice cloning request...")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Voice cloning successful!")
            print(f"Output file: {result.get('output_file')}")
        else:
            print(f"❌ Voice cloning failed: {response.status_code}")
            print(response.text)
    
    # Clean up
    if os.path.exists(test_audio_path):
        os.remove(test_audio_path)

if __name__ == "__main__":
    test_voice_cloning() 