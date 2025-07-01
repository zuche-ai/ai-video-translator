#!/usr/bin/env python3
"""
Download XTTS model locally for Docker mounting.
"""

import os
import sys
import torch

# Set environment variable to accept Coqui TTS license
os.environ['COQUI_TOS_AGREED'] = '1'

def download_xtts():
    """Download XTTS model to local models directory."""
    try:
        from TTS.api import TTS
    except ImportError:
        print("TTS not available. Please install TTS first:")
        print("pip install TTS")
        return False
    
    models_dir = "models/xtts"
    os.makedirs(models_dir, exist_ok=True)
    
    # Set TTS home to our local directory
    os.environ['TTS_HOME'] = os.path.abspath(models_dir)
    
    # Fix PyTorch 2.6 compatibility issue
    original_torch_load = torch.load
    def safe_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = safe_torch_load
    
    models_to_try = [
        "tts_models/multilingual/multi-dataset/xtts_v2",
        "tts_models/multilingual/multi-dataset/xtts_v1.1"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"Downloading {model_name} to {models_dir}...")
            tts = TTS(model_name, progress_bar=True)
            print(f"{model_name} downloaded successfully!")
            
            # Test that it works
            print("Testing model...")
            test_text = "Hello, this is a test."
            test_file = "test_output.wav"
            tts.tts_to_file(text=test_text, file_path=test_file)
            if os.path.exists(test_file):
                os.remove(test_file)
                print("Model test successful!")
            
            return True
            
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            continue
    
    print("Failed to download any XTTS model")
    return False

if __name__ == "__main__":
    success = download_xtts()
    if success:
        print(f"\nXTTS model downloaded to: {os.path.abspath('models/xtts')}")
        print("You can now mount this directory in Docker.")
    else:
        sys.exit(1) 