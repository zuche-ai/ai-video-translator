#!/usr/bin/env python3
"""
Example script demonstrating audio translation functionality.

This script shows how to use the AudioTranslator class to translate audio files
from one language to another, with options for voice cloning or regular TTS.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_translator.audio_translation.audio_translator import AudioTranslator


def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main example function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Example 1: Basic audio translation with voice cloning
    logger.info("=== Example 1: Audio Translation with Voice Cloning ===")
    
    # Check if we have a sample audio file
    sample_audio = "sample_audio.mp3"  # You would need to provide this
    
    if not os.path.exists(sample_audio):
        logger.warning(f"Sample audio file '{sample_audio}' not found. Skipping example.")
        logger.info("To test this example, place an audio file named 'sample_audio.mp3' in the examples directory.")
    else:
        try:
            # Create audio translator with voice cloning
            translator = AudioTranslator(
                src_lang="en",
                tgt_lang="es",
                model_size="base",
                voice_clone=True
            )
            
            # Translate the audio
            output_path = translator.translate_audio(
                input_path=sample_audio,
                output_path="translated_audio_es.mp3"
            )
            
            logger.info(f"Translation completed! Output: {output_path}")
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
    
    # Example 2: List supported formats
    logger.info("\n=== Example 2: Supported Audio Formats ===")
    translator = AudioTranslator()
    formats = translator.get_supported_formats()
    logger.info("Supported audio formats:")
    for fmt in formats:
        logger.info(f"  {fmt}")
    
    # Example 3: Validate input file
    logger.info("\n=== Example 3: Input File Validation ===")
    test_files = ["audio.mp3", "audio.wav", "document.txt", "image.jpg"]
    
    for file_path in test_files:
        is_valid = translator.validate_input_file(file_path)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        logger.info(f"{file_path}: {status}")
    
    # Example 4: CLI usage instructions
    logger.info("\n=== Example 4: CLI Usage ===")
    logger.info("You can also use the command-line interface:")
    logger.info("")
    logger.info("  # Basic usage with voice cloning")
    logger.info("  python -m video_translator.audio_translation.cli \\")
    logger.info("    --input audio.mp3 \\")
    logger.info("    --src-lang en \\")
    logger.info("    --tgt-lang es \\")
    logger.info("    --voice-clone")
    logger.info("")
    logger.info("  # List supported formats")
    logger.info("  python -m video_translator.audio_translation.cli --list-formats")
    logger.info("")
    logger.info("  # Verbose output")
    logger.info("  python -m video_translator.audio_translation.cli \\")
    logger.info("    --input audio.mp3 \\")
    logger.info("    --src-lang en \\")
    logger.info("    --tgt-lang es \\")
    logger.info("    --voice-clone \\")
    logger.info("    --verbose")


if __name__ == "__main__":
    main() 