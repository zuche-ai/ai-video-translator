import argparse
import sys
import os

# Import core modules
from .transcriber import transcribe_video
from .translator import translate_segments
from .subtitles import generate_srt
from .video_editor import burn_subtitles
from .voice_cloner import VoiceCloner, AudioProcessor

def main():
    parser = argparse.ArgumentParser(description="Transcribe, translate, and caption videos with translated subtitles and voice cloning.")
    parser.add_argument('--input', required=True, help='Path to input video file')
    parser.add_argument('--src-lang', required=True, help='Source language code (e.g., en, fr, de)')
    parser.add_argument('--tgt-lang', required=True, help='Target language code for captions')
    parser.add_argument('--output', required=True, help='Path for output video file')
    parser.add_argument('--voice-clone', action='store_true', help='Enable voice cloning with translated audio')
    parser.add_argument('--audio-mode', choices=['replace', 'overlay', 'subtitles-only'], 
                       default='subtitles-only', help='Audio processing mode')
    parser.add_argument('--original-volume', type=float, default=0.3, 
                       help='Volume of original audio when overlaying (0.0 to 1.0)')
    parser.add_argument('--voice-profile-duration', type=float, default=10.0,
                       help='Duration of audio to use for voice profiling (seconds)')
    parser.add_argument('--debug', action='store_true', help='Print all commands and actions')
    args = parser.parse_args()

    try:
        print(f"[1/5] Transcribing audio with Whisper...")
        segments = transcribe_video(args.input, args.src_lang, debug=args.debug)

        print(f"[2/5] Translating transcription to {args.tgt_lang}...")
        translated_segments = translate_segments(segments, args.src_lang, args.tgt_lang, debug=args.debug)

        print(f"[3/5] Generating SRT subtitles...")
        srt_path = generate_srt(translated_segments, args.input, args.tgt_lang, debug=args.debug)

        # Voice cloning and audio processing
        if args.voice_clone and args.audio_mode != 'subtitles-only':
            print(f"[4/5] Processing voice cloning and audio...")
            
            # Initialize voice cloner and audio processor
            voice_cloner = VoiceCloner()
            audio_processor = AudioProcessor()
            
            # Extract audio from video for voice profiling
            temp_audio = f"{args.input}_temp_audio.wav"
            if not audio_processor.extract_audio_from_video(args.input, temp_audio):
                raise RuntimeError("Failed to extract audio from video")
            
            # Extract voice profile
            if not voice_cloner.extract_voice_profile(temp_audio, args.voice_profile_duration):
                raise RuntimeError("Failed to extract voice profile")
            
            # Generate translated audio segments
            temp_audio_dir = f"{args.input}_translated_audio"
            os.makedirs(temp_audio_dir, exist_ok=True)
            
            texts = [seg['text'] for seg in translated_segments]
            timestamps = [seg['start'] for seg in translated_segments]
            
            audio_files = voice_cloner.batch_clone_voice(
                texts, temp_audio_dir, language=args.tgt_lang
            )
            
            if not audio_files:
                raise RuntimeError("Failed to generate translated audio")
            
            # Merge audio segments
            merged_audio = f"{args.input}_merged_audio.wav"
            if not audio_processor.merge_audio_segments(audio_files, merged_audio, timestamps):
                raise RuntimeError("Failed to merge audio segments")
            
            # Process audio based on mode
            final_audio = f"{args.input}_final_audio.wav"
            if args.audio_mode == 'replace':
                # Simply use the translated audio
                final_audio = merged_audio
            elif args.audio_mode == 'overlay':
                # Overlay translated audio on original
                if not audio_processor.overlay_audio(temp_audio, merged_audio, final_audio, args.original_volume):
                    raise RuntimeError("Failed to overlay audio")
            
            # Clean up temporary files
            voice_cloner.cleanup()
            if os.path.exists(temp_audio):
                os.unlink(temp_audio)
            
            print(f"[5/5] Burning subtitles and applying audio to video...")
            burn_subtitles(args.input, srt_path, args.output, audio_file=final_audio, debug=args.debug)
            
            # Clean up more temporary files
            if os.path.exists(merged_audio) and merged_audio != final_audio:
                os.unlink(merged_audio)
            if os.path.exists(final_audio):
                os.unlink(final_audio)
            
        else:
            print(f"[4/4] Burning subtitles into video...")
            burn_subtitles(args.input, srt_path, args.output, debug=args.debug)

        print(f"\n✅ Done! Output saved to {args.output}")
        
        if args.voice_clone and args.audio_mode != 'subtitles-only':
            print(f"🎤 Voice cloning completed with {args.audio_mode} mode")
        
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
