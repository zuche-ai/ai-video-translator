import argparse
import sys
import os
import tempfile
import numpy as np
import torch
import time

# Import core modules
from .core.transcriber import transcribe_video
from .core.translator import translate_segments
from .core.subtitles import generate_srt
from .video.video_editor import burn_subtitles
from .audio.voice_cloner import VoiceCloner

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
    parser.add_argument('--reference-audio', type=str, 
                       help='Path to reference audio file for voice cloning (if not provided, will extract from video)')
    parser.add_argument('--debug', action='store_true', help='Print all commands and actions')
    args = parser.parse_args()

    try:
        print(f"[1/5] Transcribing audio with Whisper...")
        segments = transcribe_video(args.input, args.src_lang, debug=args.debug)

        print(f"[2/5] Translating transcription to {args.tgt_lang}...")
        translated_segments = translate_segments(segments, args.src_lang, args.tgt_lang, debug=args.debug)

        # Clear memory after transcription and translation
        import gc
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Add a small delay to ensure resources are freed
        time.sleep(1)

        # Voice cloning and audio processing
        if args.voice_clone and args.audio_mode != 'subtitles-only':
            print(f"[3/5] Processing voice cloning and audio...")
            
            # Initialize voice cloner with error handling
            try:
                print("About to initialize VoiceCloner...")
                voice_cloner = VoiceCloner()
                print("Voice cloner initialized successfully")
            except Exception as e:
                print(f"Failed to initialize voice cloner: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("Falling back to subtitles-only mode...")
                args.voice_clone = False
                # Continue with subtitles-only mode
                srt_path = generate_srt(translated_segments, args.input, args.tgt_lang, debug=args.debug)
                burn_subtitles(args.input, srt_path, args.output, debug=args.debug)
                print(f"\n✅ Done! Output saved to {args.output}")
                print(f"📝 Subtitles added in {args.tgt_lang} (voice cloning failed)")
                return
            
            # Get reference audio for voice cloning
            try:
                print("Getting reference audio...")
                if args.reference_audio:
                    reference_audio = args.reference_audio
                    if not voice_cloner.validate_reference_audio(reference_audio):
                        raise RuntimeError("Invalid reference audio file")
                else:
                    # Extract audio from video for voice cloning
                    import ffmpeg
                    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    try:
                        stream = ffmpeg.input(args.input)
                        stream = ffmpeg.output(stream, temp_audio, acodec='pcm_s16le', ac=1, ar='22050')
                        ffmpeg.run(stream, overwrite_output=True, quiet=not args.debug)
                        reference_audio = temp_audio
                    except Exception as e:
                        raise RuntimeError(f"Failed to extract audio from video: {e}")
                print(f"Reference audio ready: {reference_audio}")
            except Exception as e:
                print(f"Failed to get reference audio: {e}")
                raise

            # --- Quality checks ---
            try:
                print("Running quality checks...")
                snr_ok = voice_cloner.check_snr(reference_audio)
                vad_ok = voice_cloner.check_vad(reference_audio)
                print(f"SNR check: {snr_ok}, VAD check: {vad_ok}")
                if not snr_ok or not vad_ok:
                    print("[WARNING] Reference audio is too noisy or lacks clear speech for voice cloning. Using default TTS voice.")
                    args.voice_clone = False  # Fallback to TTS
            except Exception as e:
                print(f"Quality checks failed: {e}")
                # Continue anyway

        # Voice cloning branch (only if checks passed)
        if args.voice_clone and args.audio_mode != 'subtitles-only':
            print(f"[3/5] Voice cloning checks passed, proceeding with voice cloning...")
            
            # Generate translated audio segments
            try:
                print("Creating temp audio directory...")
                temp_audio_dir = tempfile.mkdtemp()
                print(f"Created temp audio directory: {temp_audio_dir}")
                
                texts = [seg['text'] for seg in translated_segments]
                timestamps = [(seg['start'], seg['end']) for seg in translated_segments]
                
                print(f"Processing {len(texts)} text segments for voice cloning...")
                print(f"First few texts: {texts[:3]}")
                print(f"Reference audio: {reference_audio}")
                
                print("About to call batch_clone_voice...")
                audio_files = voice_cloner.batch_clone_voice(
                    reference_audio_path=reference_audio,
                    texts=texts,
                    output_dir=temp_audio_dir,
                    language=args.tgt_lang
                )
                
                if not audio_files:
                    raise RuntimeError("Failed to generate translated audio")
                    
                print(f"Successfully generated {len(audio_files)} audio files")
                
            except Exception as e:
                print(f"Voice cloning failed: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("Falling back to subtitles-only mode...")
                args.voice_clone = False
                # Continue with subtitles-only mode
                srt_path = generate_srt(translated_segments, args.input, args.tgt_lang, debug=args.debug)
                burn_subtitles(args.input, srt_path, args.output, debug=args.debug)
                print(f"\n✅ Done! Output saved to {args.output}")
                print(f"📝 Subtitles added in {args.tgt_lang} (voice cloning failed)")
                return
            
            # Create synchronized audio with proper timing
            import librosa
            import soundfile as sf
            
            # Load original audio to get total duration
            original_audio, sr = librosa.load(reference_audio, sr=22050)
            total_duration = len(original_audio) / sr
            
            # Create a timeline for the translated audio
            translated_timeline = np.zeros_like(original_audio)
            
            # Place each translated segment at its correct timestamp
            for i, (audio_file, (start_time, end_time)) in enumerate(zip(audio_files, timestamps)):
                # Load the translated audio segment
                segment_audio, segment_sr = librosa.load(audio_file, sr=22050)
                
                # Convert timestamps to sample indices
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_length = end_sample - start_sample
                
                # Resize segment to match the original timing if needed
                if len(segment_audio) != segment_length:
                    # Use librosa's time stretching to match duration
                    segment_audio = librosa.effects.time_stretch(segment_audio, rate=len(segment_audio)/segment_length)
                
                # Ensure segment fits within the timeline
                if start_sample + len(segment_audio) <= len(translated_timeline):
                    translated_timeline[start_sample:start_sample + len(segment_audio)] = segment_audio
                else:
                    # Truncate if it goes beyond the timeline
                    fit_length = len(translated_timeline) - start_sample
                    translated_timeline[start_sample:] = segment_audio[:fit_length]
            
            # Save the synchronized translated audio
            merged_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(merged_audio, translated_timeline, sr)
            
            # Process audio based on mode
            final_audio = merged_audio
            if args.audio_mode == 'overlay':
                # Overlay translated audio on original (now properly synchronized)
                # Mix audio at the correct timestamps
                mixed_audio = (original_audio * args.original_volume + 
                             translated_timeline * (1.0 - args.original_volume))
                
                final_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                sf.write(final_audio, mixed_audio, sr)
            
            print(f"[4/5] Creating video with cloned audio (no captions)...")
            # Create video with cloned audio but no captions
            burn_subtitles(args.input, None, args.output, audio_file=final_audio, debug=args.debug)
            
            # Clean up temporary files
            if 'temp_audio' in locals():
                os.unlink(temp_audio)
            if os.path.exists(merged_audio):
                os.unlink(merged_audio)
            if final_audio != merged_audio and os.path.exists(final_audio):
                os.unlink(final_audio)
            for audio_file in audio_files:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
            if os.path.exists(temp_audio_dir):
                os.rmdir(temp_audio_dir)
            
        else:
            print(f"[3/5] Generating SRT subtitles...")
            srt_path = generate_srt(translated_segments, args.input, args.tgt_lang, debug=args.debug)
            
            print(f"[4/5] Burning subtitles into video...")
            burn_subtitles(args.input, srt_path, args.output, debug=args.debug)

        print(f"\n✅ Done! Output saved to {args.output}")
        
        if args.voice_clone and args.audio_mode != 'subtitles-only':
            print(f"🎤 Voice cloning completed with {args.audio_mode} mode (no captions)")
        else:
            print(f"📝 Subtitles added in {args.tgt_lang}")
        
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
