import argparse
import sys

# Import core modules
from .transcriber import transcribe_video
from .translator import translate_segments
from .subtitles import generate_srt
from .video_editor import burn_subtitles

def main():
    parser = argparse.ArgumentParser(description="Transcribe, translate, and caption videos with translated subtitles.")
    parser.add_argument('--input', required=True, help='Path to input video file')
    parser.add_argument('--src-lang', required=True, help='Source language code (e.g., en, fr, de)')
    parser.add_argument('--tgt-lang', required=True, help='Target language code for captions')
    parser.add_argument('--output', required=True, help='Path for output video file')
    parser.add_argument('--debug', action='store_true', help='Print all commands and actions')
    args = parser.parse_args()

    try:
        print(f"[1/4] Transcribing audio with Whisper...")
        segments = transcribe_video(args.input, args.src_lang, debug=args.debug)

        print(f"[2/4] Translating transcription to {args.tgt_lang}...")
        translated_segments = translate_segments(segments, args.src_lang, args.tgt_lang, debug=args.debug)

        print(f"[3/4] Generating SRT subtitles...")
        srt_path = generate_srt(translated_segments, args.input, args.tgt_lang, debug=args.debug)

        print(f"[4/4] Burning subtitles into video...")
        burn_subtitles(args.input, srt_path, args.output, debug=args.debug)

        print(f"\n✅ Done! Output saved to {args.output}")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
