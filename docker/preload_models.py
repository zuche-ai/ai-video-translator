#!/usr/bin/env python3
"""
Pre-download XTTS models during Docker build to avoid license prompts at runtime.
"""

import os
import sys

# Set environment variable to accept Coqui TTS license
os.environ['COQUI_TOS_AGREED'] = '1'

def preload_models():
    """Pre-download XTTS models."""
    try:
        from TTS.api import TTS
    except ImportError:
        print("TTS not available, skipping model preload")
        return
    
    models_to_try = [
        "tts_models/multilingual/multi-dataset/xtts_v2",
        "tts_models/multilingual/multi-dataset/xtts_v1.1"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"Pre-downloading {model_name}...")
            TTS(model_name, progress_bar=False)
            print(f"{model_name} downloaded successfully")
            break  # Success, no need to try others
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            continue
    
    print("Model preload completed")

if __name__ == "__main__":
    preload_models() 