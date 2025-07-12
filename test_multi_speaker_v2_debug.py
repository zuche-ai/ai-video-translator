#!/usr/bin/env python3
"""
Debug test script for Multi-Speaker Audio Translator V2

This script tests the improved multi-speaker translation approach with:
- Original Whisper segments (no merging) to preserve speaker separation
- Individual audio segment extraction for each Whisper segment
- Voice cloning with unique audio references per segment
- Better non-voice segment detection (intro music, gaps, etc.)
- Proper audio reconstruction with timing
"""

import os
import sys
import logging
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_translator.audio_translation.multi_speaker_translator_v2 import translate_multi_speaker_audio_v2

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
    logger.info("Starting Multi-Speaker Audio Translator V2 Debug Test")
    
    # Test file path
    test_audio = "christtestshort.wav"
    
    # Check if test file exists
    if not os.path.exists(test_audio):
        logger.error(f"Test file not found: {test_audio}")
        logger.info("Please ensure christtestshort.wav is in the current directory")
        return
    
    logger.info(f"Using test file: {test_audio}")
    
    # Output path
    output_path = "translated_multi_speaker_v2_christtestshort.wav"
    
    try:
        # Start timing
        start_time = time.time()
        
        logger.info("Starting multi-speaker translation V2...")
        
        # Run the translation
        result_path = translate_multi_speaker_audio_v2(
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
                logger.info("✅ SUCCESS: Multi-speaker translation V2 completed successfully!")
                logger.info(f"🎵 Output audio: {result_path}")
                logger.info("🎯 Key improvements:")
                logger.info("   - Uses original Whisper segments (no merging)")
                logger.info("   - Each segment gets its own audio reference")
                logger.info("   - Better non-voice segment detection")
                logger.info("   - Preserves speaker separation")
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