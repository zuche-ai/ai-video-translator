"""
Command-line interface for audio translation.
"""

import argparse
import logging
import sys
from pathlib import Path

from .audio_translator import AudioTranslator


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Translate audio files to different languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic audio translation with voice cloning
  python -m video_translator.audio_translation.cli --input audio.mp3 --src-lang en --tgt-lang es --voice-clone
  
  # Audio translation with regular TTS (when implemented)
  python -m video_translator.audio_translation.cli --input audio.wav --src-lang en --tgt-lang fr
  
  # Specify output path
  python -m video_translator.audio_translation.cli --input audio.mp3 --output translated_audio.mp3 --src-lang en --tgt-lang es --voice-clone
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=False,
        help="Input audio file path"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        help="Output audio file path (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--src-lang",
        default="en",
        help="Source language code (default: en)"
    )
    
    parser.add_argument(
        "--tgt-lang",
        default="es",
        help="Target language code (default: es)"
    )
    
    parser.add_argument(
        "--model-size",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)"
    )
    
    parser.add_argument(
        "--voice-clone",
        action="store_true",
        help="Use voice cloning instead of regular TTS"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List supported audio formats and exit"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # List supported formats
    if args.list_formats:
        translator = AudioTranslator()
        formats = translator.get_supported_formats()
        print("Supported audio formats:")
        for fmt in formats:
            print(f"  {fmt}")
        return
    
    # Check if input is required (not needed for --list-formats)
    if not args.list_formats and not args.input:
        logger.error("Input file is required unless using --list-formats")
        sys.exit(1)
    
    # Validate input file (only if not listing formats)
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        translator = AudioTranslator(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            model_size=args.model_size,
            voice_clone=args.voice_clone
        )
        
        if not translator.validate_input_file(str(input_path)):
            logger.error(f"Unsupported audio format: {input_path.suffix}")
            logger.error(f"Supported formats: {translator.get_supported_formats()}")
            sys.exit(1)
    else:
        translator = AudioTranslator()
    
    try:
        if args.input:
            logger.info("Starting audio translation...")
            output_path = translator.translate_audio(
                input_path=str(input_path),
                output_path=args.output,
                voice_clone=args.voice_clone
            )
            
            logger.info(f"Audio translation completed successfully!")
            logger.info(f"Output file: {output_path}")
        
    except Exception as e:
        logger.error(f"Audio translation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 