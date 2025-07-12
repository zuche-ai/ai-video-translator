#!/usr/bin/env python3
"""
Debug test script for Multi-Speaker Audio Translator

This script tests the new multi-speaker translation approach with:
- Sentence merging from Whisper segments
- Audio segment extraction for voice references
- Voice cloning with individual audio references
- Non-voice segment handling
- Audio reconstruction
"""

import os
import sys
import logging
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_translator.audio_translation.multi_speaker_translator import translate_multi_speaker_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def progress_callback(current_step: int, total_steps: int, message: str):
    """Progress callback function."""
    percentage = (current_step / total_steps) * 100 if total_steps > 0 else 0
    logger.info(f"Progress: {percentage:.1f}% - {message}")


def main():
    """Main test function."""
    logger.info("Starting Multi-Speaker Audio Translator Debug Test")
    
    # Test file path
    test_audio = "christtestshort.wav"
    
    # Check if test file exists
    if not os.path.exists(test_audio):
        logger.error(f"Test file not found: {test_audio}")
        logger.info("Please ensure christtestshort.wav is in the current directory")
        return
    
    logger.info(f"Using test file: {test_audio}")
    
    # Output path
    output_path = "translated_multi_speaker_christtestshort.wav"
    
    try:
        # Start timing
        start_time = time.time()
        
        logger.info("Starting multi-speaker translation...")
        
        # Run the translation
        result_path = translate_multi_speaker_audio(
            audio_path=test_audio,
            src_lang="en",
            tgt_lang="es",
            output_path=output_path,
            progress_hook=progress_callback
        )
        
        # End timing
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Translation completed successfully!")
        logger.info(f"Output file: {result_path}")
        logger.info(f"Total time: {duration:.2f} seconds")
        
        # Check if output file exists and has content
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path)
            logger.info(f"Output file size: {file_size} bytes")
            
            if file_size > 0:
                logger.info("✅ SUCCESS: Multi-speaker translation completed successfully!")
                logger.info(f"🎵 Output audio: {result_path}")
            else:
                logger.error("❌ ERROR: Output file is empty")
        else:
            logger.error("❌ ERROR: Output file was not created")
            
    except Exception as e:
        logger.error(f"❌ ERROR: Translation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 