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

def process(input_path, output_path, src_lang, tgt_lang, voice_clone, audio_mode, original_volume, add_captions, caption_font_size, debug=False, progress_hook=None):
    try:
        print(f"[MAIN] Starting process function...")
        print(f"[MAIN] Input path: {input_path}")
        print(f"[MAIN] Output path: {output_path}")
        print(f"[MAIN] Source language: {src_lang}")
        print(f"[MAIN] Target language: {tgt_lang}")
        print(f"[MAIN] Voice clone: {voice_clone}")
        print(f"[MAIN] Audio mode: {audio_mode}")
        
        print(f"[1/5] Transcribing audio with Whisper...")
        print(f"[MAIN] About to call transcribe_video...")
        segments = transcribe_video(input_path, src_lang, debug=debug, timeout=600)  # 10 minute timeout
        print(f"[MAIN] transcribe_video completed successfully")
        
        # Check if transcription found any speech
        if not segments:
            print("⚠️ No speech detected in video. Creating video with subtitles only.")
            # Create a simple video with a message
            srt_path = generate_srt([{
                'start': 0.0,
                'end': 5.0,
                'text': 'No speech detected in this video'
            }], input_path, tgt_lang, debug=debug)
            burn_subtitles(input_path, srt_path, output_path, caption_font_size=caption_font_size, debug=debug)
            if progress_hook:
                progress_hook(3)
            if progress_hook:
                progress_hook(4)
            return
        
        if progress_hook:
            progress_hook(1)

        print(f"[2/5] Translating transcription to {tgt_lang}...")
        translated_segments = translate_segments(segments, src_lang, tgt_lang, debug=debug)
        if progress_hook:
            progress_hook(2)

        # Clear memory after transcription and translation
        import gc
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)

        # Voice cloning and audio processing
        if voice_clone and audio_mode != 'subtitles-only':
            print(f"[3/5] Processing voice cloning and audio...")
            try:
                print("About to initialize VoiceCloner...")
                voice_cloner = VoiceCloner()
                print("Voice cloner initialized successfully")
                
                # Check if TTS model was actually loaded
                if voice_cloner.tts is None:
                    raise RuntimeError("TTS model failed to load")
                    
            except Exception as e:
                print(f"Failed to initialize voice cloner: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Voice cloning initialization failed: {e}")
            # Extract reference audio from video
            import ffmpeg
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            try:
                stream = ffmpeg.input(input_path)
                stream = ffmpeg.output(stream, temp_audio, acodec='pcm_s16le', ac=1, ar='22050')
                ffmpeg.run(stream, overwrite_output=True, quiet=not debug)
                reference_audio = temp_audio
            except Exception as e:
                raise RuntimeError(f"Failed to extract audio from video: {e}")
            # Quality checks
            try:
                snr_ok = voice_cloner.check_snr(reference_audio)
                vad_ok = voice_cloner.check_vad(reference_audio)
                if not snr_ok or not vad_ok:
                    print("[WARNING] Reference audio is too noisy or lacks clear speech for voice cloning. Using default TTS voice.")
                    voice_clone = False
            except Exception as e:
                print(f"Quality checks failed: {e}")
            if voice_clone:
                # Generate translated audio segments
                temp_audio_dir = tempfile.mkdtemp()
                texts = [seg['text'] for seg in translated_segments]
                timestamps = [(seg['start'], seg['end']) for seg in translated_segments]
                audio_files = voice_cloner.batch_clone_voice(
                    reference_audio_path=reference_audio,
                    texts=texts,
                    output_dir=temp_audio_dir,
                    language=tgt_lang
                )
                import librosa
                import soundfile as sf
                original_audio, sr = librosa.load(reference_audio, sr=22050)
                translated_timeline = np.zeros_like(original_audio)
                for i, (audio_file, (start_time, end_time)) in enumerate(zip(audio_files, timestamps)):
                    segment_audio, segment_sr = librosa.load(audio_file, sr=22050)
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segment_length = end_sample - start_sample
                    if len(segment_audio) != segment_length and segment_length > 0:
                        segment_audio = librosa.effects.time_stretch(segment_audio, rate=len(segment_audio)/segment_length)
                    if start_sample + len(segment_audio) <= len(translated_timeline):
                        translated_timeline[start_sample:start_sample + len(segment_audio)] = segment_audio
                    else:
                        fit_length = len(translated_timeline) - start_sample
                        translated_timeline[start_sample:] = segment_audio[:fit_length]
                merged_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                sf.write(merged_audio, translated_timeline, sr)
                final_audio = merged_audio
                if audio_mode == 'overlay':
                    mixed_audio = (original_audio * original_volume + translated_timeline * (1.0 - original_volume))
                    final_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    sf.write(final_audio, mixed_audio, sr)
                srt_path = None
                if add_captions:
                    srt_path = generate_srt(translated_segments, input_path, tgt_lang, debug=debug)
                burn_subtitles(input_path, srt_path, output_path, audio_file=final_audio, caption_font_size=caption_font_size, debug=debug)
                if progress_hook:
                    progress_hook(3)
                if progress_hook:
                    progress_hook(4)
                # Clean up temp files
                if os.path.exists(merged_audio):
                    os.unlink(merged_audio)
                if final_audio != merged_audio and os.path.exists(final_audio):
                    os.unlink(final_audio)
                for audio_file in audio_files:
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
                if os.path.exists(temp_audio_dir):
                    os.rmdir(temp_audio_dir)
                if os.path.exists(temp_audio):
                    os.unlink(temp_audio)
                return
        # Subtitles-only or fallback
        print(f"[3/5] Generating SRT subtitles...")
        srt_path = generate_srt(translated_segments, input_path, tgt_lang, debug=debug)
        if progress_hook:
            progress_hook(3)
        print(f"[4/5] Burning subtitles into video...")
        burn_subtitles(input_path, srt_path, output_path, caption_font_size=caption_font_size, debug=debug)
        if progress_hook:
            progress_hook(4)
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

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
    parser.add_argument('--add-captions', action='store_true', help='Burn subtitles even when using overlay or replace audio modes')
    parser.add_argument('--caption-font-size', type=int, default=24, help='Font size for subtitles (default: 24)')
    args = parser.parse_args()
    try:
        process(
            input_path=args.input,
            output_path=args.output,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            voice_clone=args.voice_clone,
            audio_mode=args.audio_mode,
            original_volume=args.original_volume,
            add_captions=args.add_captions,
            caption_font_size=args.caption_font_size,
            debug=args.debug
        )
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
