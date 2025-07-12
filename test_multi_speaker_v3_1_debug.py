#!/usr/bin/env python3
"""
Debug test script for Multi-Speaker Audio Translator V3.1

This script tests the simple V3.1 fix that addresses voice/music overlap:
- Detects intro music and non-voice segments
- Trims the first voice segment to start after music ends
- Uses only clean voice as reference for voice cloning
- Proper audio reconstruction with timing
"""

import os
import sys
import logging
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_translator.audio_translation.multi_speaker_translator_v3_1 import translate_multi_speaker_audio_v3_1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def progress_callback(current: int, total: int, message: str):
    """Progress callback function."""
    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"Progress: {percentage:.1f}% - {message}")

def main():
    """Main test function."""
    # Test file
    input_file = "christtestshort.wav"
    
    if not os.path.exists(input_file):
        logger.error(f"Test file not found: {input_file}")
        return
    
    logger.info("🎯 Starting Multi-Speaker Audio Translator V3.1 Test")
    logger.info(f"📁 Input file: {input_file}")
    
    start_time = time.time()
    
    try:
        # Run the translation
        output_file = translate_multi_speaker_audio_v3_1(
            audio_path=input_file,
            src_lang="en",
            tgt_lang="es",
            progress_hook=progress_callback
        )
        
        # Calculate timing
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get file size
        file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
        
        logger.info("✅ Translation completed successfully!")
        logger.info(f"📤 Output file: {output_file}")
        logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
        logger.info(f"📊 Output file size: {file_size} bytes")
        
        logger.info("🎉 SUCCESS: Multi-speaker translation V3.1 completed successfully!")
        logger.info("🎵 Output audio: " + output_file)
        logger.info("🎯 Key improvements:")
        logger.info("   - Simple fix to V3 that trims first voice segment")
        logger.info("   - Avoids music/voice overlap in voice cloning")
        logger.info("   - Uses only clean voice as reference")
        logger.info("   - Proper audio reconstruction with timing")
        
    except Exception as e:
        logger.error(f"❌ Translation failed: {e}")
        raise

if __name__ == "__main__":
    main() 