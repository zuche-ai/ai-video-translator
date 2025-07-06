import argparse
import sys
import os
import tempfile
import numpy as np
import torch
import time
import logging
import shutil

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
        
        # --- Granular Progress: 0-10% ---
        if progress_hook:
            progress_hook(1, 0, "Transcribing audio (loading model)")
        print(f"[1/5] Transcribing audio with Whisper...")
        print(f"[MAIN] About to call transcribe_video...")
        segments = transcribe_video(input_path, src_lang, debug=debug, timeout=600)  # 10 minute timeout
        if progress_hook:
            progress_hook(1, 10, "Transcribing audio (running inference)")
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
                progress_hook(4, 90, "Finalizing (burning subtitles)")
                progress_hook(5, 100, "Done")
            return
        
        # --- Granular Progress: 10-40% (Translation) ---
        if progress_hook:
            progress_hook(2, 10, f"Translating {len(segments)} segments")
        print(f"[2/5] Translating transcription to {tgt_lang}...")
        translated_segments = []
        total_segments = len(segments)
        for i, segment in enumerate(segments):
            translated = translate_segments([segment], src_lang, tgt_lang, debug=debug)
            translated_segments.extend(translated)
            if progress_hook:
                percent = 10 + int(((i+1)/total_segments)*30)
                progress_hook(2, percent, f"Translating segment {i+1}/{total_segments}")
        
        # Clear memory after transcription and translation
        import gc
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)
        
        # --- Granular Progress: 40-80% (Voice cloning/audio) ---
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
                audio_files = []
                total_audio = len(texts)
                for i, text in enumerate(texts):
                    audio_file = voice_cloner.clone_voice(
                        text=text,
                        audio_path=reference_audio,
                        output_path=os.path.join(temp_audio_dir, f"cloned_audio_{i:04d}.wav"),
                        language=tgt_lang
                    )
                    audio_files.append(audio_file)
                    if progress_hook:
                        percent = 40 + int(((i+1)/total_audio)*40)
                        progress_hook(3, percent, f"Voice cloning segment {i+1}/{total_audio}")
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
                    progress_hook(4, 90, "Finalizing (burning subtitles)")
                    progress_hook(5, 100, "Done")
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
            progress_hook(4, 90, "Finalizing (burning subtitles)")
        print(f"[4/5] Burning subtitles into video...")
        burn_subtitles(input_path, srt_path, output_path, caption_font_size=caption_font_size, debug=debug)
        if progress_hook:
            progress_hook(5, 100, "Done")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def process_audio_file(input_path, output_path, src_lang, tgt_lang, voice_clone, debug=False):
    """
    Pipeline for audio-only translation with voice cloning, non-speech event handling, and silence trimming.
    Steps:
      1. Use Silero VAD to segment speech/non-speech
      2. Transcribe speech segments with Whisper
      3. Translate with transformers
      4. Synthesize with Coqui XTTS (voice cloning)
      5. Splice original audio for non-speech events
      6. Trim long silences
      7. Save output audio
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("audio_pipeline")
    
    # Add print statements for Docker logs
    print(f"[DEBUG] Starting process_audio_file")
    print(f"[DEBUG] Input: {input_path}, Output: {output_path}")
    print(f"[DEBUG] Source lang: {src_lang}, Target lang: {tgt_lang}, Voice clone: {voice_clone}")
    
    logger.info(f"[AUDIO_PIPELINE] Input: {input_path}, Output: {output_path}")
    logger.info(f"[AUDIO_PIPELINE] Source lang: {src_lang}, Target lang: {tgt_lang}, Voice clone: {voice_clone}")
    
    # Step 1: Silero VAD segmentation
    import torch
    import soundfile as sf
    import numpy as np
    import tempfile
    from silero_vad import get_speech_timestamps

    # Load audio
    audio, sr = sf.read(input_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert to mono if needed
    print(f"[DEBUG] Loaded audio: shape={audio.shape}, sr={sr}, duration={len(audio)/sr:.2f}s")
    logger.info(f"[AUDIO_PIPELINE] Loaded audio: {input_path}, shape: {audio.shape}, sr: {sr}")

    # Prepare Silero VAD model (robust to both tuple and single-object returns)
    try:
        vad_model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    except Exception:
        vad_model = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)

    # Convert audio to torch.Tensor for Silero VAD
    audio_tensor = torch.from_numpy(audio).float()

    # Run VAD
    speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sr)
    print(f"[DEBUG] Silero VAD detected {len(speech_timestamps)} speech segments")
    logger.info(f"[AUDIO_PIPELINE] Detected {len(speech_timestamps)} speech segments with Silero VAD.")
    
    if len(speech_timestamps) == 0:
        print("[DEBUG] NO SPEECH DETECTED! Trying with more lenient VAD settings...")
        # Try with more lenient settings
        try:
            speech_timestamps = get_speech_timestamps(
                audio_tensor, vad_model, sampling_rate=sr,
                threshold=0.3,  # Lower threshold (default is 0.5)
                min_speech_duration_ms=100,  # Shorter minimum speech duration
                max_speech_duration_s=float('inf')
            )
            print(f"[DEBUG] Lenient VAD detected {len(speech_timestamps)} speech segments")
        except Exception as e:
            print(f"[DEBUG] Lenient VAD failed: {e}")
        
        if len(speech_timestamps) == 0:
            print("[DEBUG] STILL NO SPEECH DETECTED! Processing entire audio as speech for testing.")
            # For testing, treat the entire audio as speech
            speech_timestamps = [{'start': 0, 'end': len(audio)}]
            print("[DEBUG] Treating entire audio as speech segment for testing")
    
    for i, seg in enumerate(speech_timestamps):
        print(f"[DEBUG] Speech segment {i}: start={seg['start']}, end={seg['end']}")
        logger.info(f"[AUDIO_PIPELINE] Speech segment {i}: start={seg['start']}, end={seg['end']}")
    
    # Detect non-speech segments as complement
    non_speech_segments = []
    last_end = 0
    for seg in speech_timestamps:
        if seg['start'] > last_end:
            non_speech_segments.append({'start': last_end, 'end': seg['start']})
        last_end = seg['end']
    if last_end < len(audio):
        non_speech_segments.append({'start': last_end, 'end': len(audio)})
    print(f"[DEBUG] Detected {len(non_speech_segments)} non-speech segments")
    logger.info(f"[AUDIO_PIPELINE] Detected {len(non_speech_segments)} non-speech segments.")
    for i, seg in enumerate(non_speech_segments):
        print(f"[DEBUG] Non-speech segment {i}: start={seg['start']}, end={seg['end']}")
        logger.info(f"[AUDIO_PIPELINE] Non-speech segment {i}: start={seg['start']}, end={seg['end']}")

    # Step 2: Whisper transcription for speech segments
    import whisper
    model = whisper.load_model('base')
    print("[DEBUG] Loaded Whisper model (base)")
    logger.info("[AUDIO_PIPELINE] Loaded Whisper model (base)")
    transcriptions = []
    for i, seg in enumerate(speech_timestamps):
        start_sample = seg['start']
        end_sample = seg['end']
        chunk = audio[start_sample:end_sample]
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_wav.name, chunk, sr)
        print(f"[DEBUG] Transcribing speech segment {i} (samples {start_sample}-{end_sample})")
        logger.info(f"[AUDIO_PIPELINE] Transcribing speech segment {i} (samples {start_sample}-{end_sample})")
        result = model.transcribe(temp_wav.name, language=src_lang)
        print(f"[DEBUG] Whisper result for segment {i}: '{result['text']}'")
        logger.info(f"[AUDIO_PIPELINE] Whisper result for segment {i}: {result['text']}")
        transcriptions.append({
            'start': start_sample / sr,
            'end': end_sample / sr,
            'text': result['text']
        })
        temp_wav.close()
    
    if len(transcriptions) == 0:
        print("[DEBUG] NO TRANSCRIPTIONS! Creating output with original audio only.")
        sf.write(output_path, audio, sr)
        return

    # Step 3: Translation for each transcription
    try:
        from transformers import pipeline
    except ImportError as e:
        print("transformers package is not installed. Please check your Docker image or requirements.txt.")
        raise e
    translator = pipeline('translation_en_to_es', model='Helsinki-NLP/opus-mt-en-es')
    print("[DEBUG] Loaded translation pipeline (en->es)")
    logger.info("[AUDIO_PIPELINE] Loaded translation pipeline (en->es)")
    translated_segments = []
    for i, seg in enumerate(transcriptions):
        translated = translator(seg['text'])[0]['translation_text']
        print(f"[DEBUG] Translation for segment {i}: '{translated}'")
        logger.info(f"[AUDIO_PIPELINE] Translation for segment {i}: {translated}")
        translated_segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': translated
        })
    
    if len(translated_segments) == 0:
        print("[DEBUG] NO TRANSLATIONS! Creating output with original audio only.")
        sf.write(output_path, audio, sr)
        return

    # Step 4: TTS/Voice Cloning for translated segments
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("[DEBUG] Initialized Coqui XTTS directly")
    logger.info("[AUDIO_PIPELINE] Initialized Coqui XTTS directly")
    tts_audio_segments = []
    for i, seg in enumerate(translated_segments):
        text = seg['text']
        start = seg['start']
        end = seg['end']
        temp_tts_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        print(f"[DEBUG] Synthesizing TTS for segment {i}: '{text}' ({start:.2f}-{end:.2f}s)")
        logger.info(f"[AUDIO_PIPELINE] Synthesizing TTS for segment {i}: '{text}' ({start:.2f}-{end:.2f}s)")
        try:
            if voice_clone:
                # Use voice cloning with reference audio
                tts.tts_to_file(
                    text=text,
                    file_path=temp_tts_wav.name,
                    speaker_wav=input_path,
                    language=tgt_lang
                )
            else:
                # Use default voice
                tts.tts_to_file(
                    text=text,
                    file_path=temp_tts_wav.name,
                    language=tgt_lang
                )
            tts_audio_segments.append({
                'start': start,
                'end': end,
                'path': temp_tts_wav.name
            })
            print(f"[DEBUG] TTS synthesis completed for segment {i}")
            logger.info(f"[AUDIO_PIPELINE] TTS synthesis completed for segment {i}")
        except Exception as e:
            print(f"[DEBUG] TTS synthesis failed for segment {i}: {e}")
            logger.error(f"[AUDIO_PIPELINE] TTS synthesis failed for segment {i}: {e}")
            raise
    
    if len(tts_audio_segments) == 0:
        print("[DEBUG] NO TTS SEGMENTS! Creating output with original audio only.")
        sf.write(output_path, audio, sr)
        return

    # Step 5: Splicing TTS segments with buffer spacing (simplified approach)
    import librosa
    import soundfile as sf
    print("[DEBUG] Splicing TTS segments with buffer spacing...")
    logger.info("[AUDIO_PIPELINE] Splicing TTS segments with buffer spacing...")
    
    # Calculate buffer spacing (0.3 seconds between segments)
    buffer_samples = int(0.3 * sr)
    print(f"[DEBUG] Using buffer spacing of {buffer_samples} samples ({0.3}s)")
    logger.info(f"[AUDIO_PIPELINE] Using buffer spacing of {buffer_samples} samples ({0.3}s)")
    
    # Calculate total output length needed
    total_tts_duration = sum([librosa.get_duration(path=seg['path']) for seg in tts_audio_segments])
    buffer_duration = (len(tts_audio_segments) - 1) * 0.3  # Buffer between segments (not before first)
    estimated_duration = total_tts_duration + buffer_duration
    output_length = int(estimated_duration * sr)
    output_audio = np.zeros(output_length)
    sr_int = int(sr)
    
    current_position = 0
    
    # Place TTS segments sequentially with buffer spacing
    for i, seg in enumerate(tts_audio_segments):
        tts_audio, tts_sr = librosa.load(seg['path'], sr=sr_int)
        
        # Add buffer before TTS segment (except for first segment)
        if i > 0:
            current_position += buffer_samples
            print(f"[DEBUG] Added {buffer_samples} sample buffer before TTS segment {i}")
            logger.info(f"[AUDIO_PIPELINE] Added {buffer_samples} sample buffer before TTS segment {i}")
        
        start_sample = current_position
        end_sample = start_sample + len(tts_audio)
        
        print(f"[DEBUG] Placing TTS segment {i}: samples {start_sample}-{end_sample}")
        logger.info(f"[AUDIO_PIPELINE] Placing TTS segment {i}: samples {start_sample}-{end_sample}")
        
        # Ensure we don't exceed output array bounds
        if end_sample <= len(output_audio):
            output_audio[start_sample:end_sample] = tts_audio
        else:
            # Truncate if necessary
            max_length = len(output_audio) - start_sample
            output_audio[start_sample:start_sample + max_length] = tts_audio[:max_length]
        
        current_position = end_sample
    
    # Save the spliced audio before silence trimming
    sf.write(output_path, output_audio, sr_int)
    print(f"[DEBUG] Spliced audio saved to {output_path}")
    logger.info(f"[AUDIO_PIPELINE] Spliced audio saved to {output_path}")
    
    # Step 6: Trim long silences in output_audio (with error handling)
    print("[DEBUG] Trimming long silences in output audio...")
    logger.info("[AUDIO_PIPELINE] Trimming long silences in output audio...")
    
    try:
        # For very long files, skip trimming to avoid memory issues
        if len(output_audio) > 10 * sr_int * 60:  # Skip trimming for files > 10 minutes
            print("[DEBUG] File too long for trimming, skipping silence removal")
            logger.info("[AUDIO_PIPELINE] File too long for trimming, skipping silence removal")
            final_audio = output_audio
        else:
            non_silent_intervals = librosa.effects.split(output_audio, top_db=30)
            min_silence_len = int(0.2 * sr_int)  # Minimum silence to keep (0.2s)
            trimmed_audio = []
            
            # Limit the number of intervals to process to prevent memory issues
            max_intervals = 50
            if len(non_silent_intervals) > max_intervals:
                print(f"[DEBUG] Too many intervals ({len(non_silent_intervals)}), limiting to {max_intervals}")
                logger.info(f"[AUDIO_PIPELINE] Too many intervals ({len(non_silent_intervals)}), limiting to {max_intervals}")
                non_silent_intervals = non_silent_intervals[:max_intervals]
            
            for i, (start, end) in enumerate(non_silent_intervals):
                print(f"[DEBUG] Keeping non-silent interval {i}: samples {start}-{end}")
                logger.info(f"[AUDIO_PIPELINE] Keeping non-silent interval {i}: samples {start}-{end}")
                trimmed_audio.append(output_audio[start:end])
                # Add a short crossfade (20ms) between segments
                if i > 0:
                    fade_len = int(0.02 * sr_int)
                    if fade_len > 0 and len(trimmed_audio[-2]) > fade_len and len(trimmed_audio[-1]) > fade_len:
                        crossfade = np.linspace(0, 1, fade_len)
                        trimmed_audio[-2][-fade_len:] = (
                            trimmed_audio[-2][-fade_len:] * (1 - crossfade) + trimmed_audio[-1][:fade_len] * crossfade
                        )
            
            if trimmed_audio:
                final_audio = np.concatenate(trimmed_audio)
            else:
                final_audio = output_audio
                
    except Exception as e:
        print(f"[DEBUG] Silence trimming failed: {e}, using unprocessed audio")
        logger.warning(f"[AUDIO_PIPELINE] Silence trimming failed: {e}, using unprocessed audio")
        final_audio = output_audio
    
    sf.write(output_path, final_audio, sr_int)
    print(f"[DEBUG] Final audio saved to {output_path}")
    logger.info(f"[AUDIO_PIPELINE] Final audio saved to {output_path}")
    # Clean up temp files
    for seg in tts_audio_segments:
        try:
            os.remove(seg['path'])
        except Exception:
            pass

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
