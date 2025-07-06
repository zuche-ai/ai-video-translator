from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import threading
import uuid
import os
import tempfile
import traceback
import sys
from flask_cors import CORS

# Import your main pipeline logic (refactor main.py if needed)
from video_translator.main import process

app = Flask(__name__)
CORS(app)

# In-memory job store: {job_id: {status, progress, result_path, error, cancel_flag}}
jobs = {}
job_threads = {}

UPLOAD_DIR = tempfile.gettempdir()
RESULTS_DIR = tempfile.gettempdir()


def run_pipeline(job_id, input_path, options):
    print(f"[API] ===== THREAD STARTED FOR JOB {job_id} =====")
    print(f"[API] Thread ID: {threading.get_ident()}")
    print(f"[API] Current working directory: {os.getcwd()}")
    print(f"[API] Python path: {sys.path}")
    
    try:
        print(f"[API] Starting pipeline for job {job_id}")
        print(f"[API] Input path: {input_path}")
        print(f"[API] Options: {options}")
        
        # Test import
        print(f"[API] Testing imports...")
        try:
            from video_translator.main import process
            print(f"[API] Import successful")
        except Exception as import_error:
            print(f"[API] Import failed: {import_error}")
            print(f"[API] Import traceback: {traceback.format_exc()}")
            raise
        
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 0
        jobs[job_id]['status_message'] = 'Starting...'
        jobs[job_id]['cancel_flag'] = False
        print(f"[API] Job status set to processing, progress 0")
        
        # Prepare output path with descriptive name
        orig_filename = jobs[job_id].get('orig_filename', 'video')
        base, _ = os.path.splitext(orig_filename)
        output_path = os.path.join(
            RESULTS_DIR,
            f"{base}_{options['src_lang']}_{options['tgt_lang']}_translated_{job_id}.mp4"
        )
        print(f"[API] Output path: {output_path}")
        
        # Step-based progress updates
        def progress_hook(step, percent, status_message=None):
            if jobs[job_id].get('cancel_flag'):
                print(f"[API] Job {job_id} cancelled, aborting progress updates.")
                raise Exception('Job cancelled by user')
            print(f"[API] Progress hook called with step {step}, percent {percent}, status '{status_message}'")
            jobs[job_id]['progress'] = percent
            if status_message is not None:
                jobs[job_id]['status_message'] = status_message
            print(f"[API] Progress set to {percent}%, status_message: {jobs[job_id].get('status_message')}")
        
        print(f"[API] About to call process function...")
        # Call the main process pipeline, injecting progress updates at key steps
        process(
            input_path=input_path,
            output_path=output_path,
            src_lang=options['src_lang'],
            tgt_lang=options['tgt_lang'],
            voice_clone=options['voice_clone'],
            audio_mode=options['audio_mode'],
            original_volume=options['original_volume'],
            add_captions=options['add_captions'],
            caption_font_size=options['caption_font_size'],
            debug=False,
            progress_hook=progress_hook
        )
        print(f"[API] Process function completed successfully")
        jobs[job_id]['status'] = 'done'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['result_path'] = output_path
        print(f"[API] Job completed successfully")
    except Exception as e:
        if str(e) == 'Job cancelled by user':
            jobs[job_id]['status'] = 'cancelled'
            jobs[job_id]['status_message'] = 'Cancelled by user'
            jobs[job_id]['error'] = 'Job cancelled by user'
            print(f"[API] Job {job_id} cancelled by user.")
        else:
            print(f"[API] Error in pipeline: {e}")
            print(f"[API] Traceback: {traceback.format_exc()}")
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['traceback'] = traceback.format_exc()
    finally:
        print(f"[API] ===== THREAD ENDING FOR JOB {job_id} =====")

@app.route('/process', methods=['POST'])
def process_route():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    video = request.files['video']
    filename = video.filename or 'uploaded_video.mp4'
    filename = secure_filename(filename)
    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")
    video.save(input_path)

    # Parse options from form data
    options = {
        'src_lang': request.form.get('src_lang', 'en'),
        'tgt_lang': request.form.get('tgt_lang', 'es'),
        'voice_clone': request.form.get('voice_clone', 'false') == 'true',
        'audio_mode': request.form.get('audio_mode', 'subtitles-only'),
        'original_volume': float(request.form.get('original_volume', 0.3)),
        'add_captions': request.form.get('add_captions', 'false') == 'true',
        'caption_font_size': int(request.form.get('caption_font_size', 24)),
    }

    job_id = str(uuid.uuid4())
    print(f"[API] Creating job {job_id} for file {filename}")
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'result_path': None,
        'error': None,
        'orig_filename': filename
    }
    print(f"[API] Job created, about to start thread...")
    print(f"[API] Thread target: {run_pipeline}")
    print(f"[API] Thread args: {job_id}, {input_path}, {options}")
    thread = threading.Thread(target=run_pipeline, args=(job_id, input_path, options))
    print(f"[API] Thread object created: {thread}")
    print(f"[API] Thread is alive before start: {thread.is_alive()}")
    print(f"[API] Thread created, starting...")
    thread.start()
    job_threads[job_id] = thread
    print(f"[API] Thread started successfully")
    print(f"[API] Thread is alive after start: {thread.is_alive()}")
    print(f"[API] Thread name: {thread.name}")
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({k: v for k, v in job.items() if k != 'result_path'})

@app.route('/result/<job_id>', methods=['GET'])
def result(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'done' or not job['result_path']:
        return jsonify({'error': 'Result not available'}), 404
    return send_file(job['result_path'], as_attachment=True)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'jobs': len(jobs)})

@app.route('/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job['status'] in ['done', 'error', 'cancelled']:
        return jsonify({'error': f'Job already {job["status"]}'}), 400
    job['cancel_flag'] = True
    job['status'] = 'cancelling'
    job['status_message'] = 'Cancelling...'
    return jsonify({'message': 'Cancellation requested'})

@app.route('/translate-audio', methods=['POST'])
def translate_audio_route():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    audio = request.files['audio']
    filename = audio.filename or 'uploaded_audio.wav'
    filename = secure_filename(filename)
    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")
    audio.save(input_path)

    # Parse options from form data (can be extended as needed)
    options = {
        'src_lang': request.form.get('src_lang', 'en'),
        'tgt_lang': request.form.get('tgt_lang', 'es'),
        'voice_clone': request.form.get('voice_clone', 'false') == 'true',
    }

    # Prepare output path
    base, _ = os.path.splitext(filename)
    output_path = os.path.join(
        RESULTS_DIR,
        f"{base}_{options['src_lang']}_{options['tgt_lang']}_translated_{uuid.uuid4()}.wav"
    )

    try:
        from video_translator.main import process_audio_file
        process_audio_file(
            input_path=input_path,
            output_path=output_path,
            src_lang=options['src_lang'],
            tgt_lang=options['tgt_lang'],
            voice_clone=options['voice_clone'],
            debug=False
        )
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        print(f"[API] Error in audio translation: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe-and-translate', methods=['POST'])
def transcribe_and_translate_route():
    """
    Step 1: Transcribe and translate audio to SRT files for human review.
    Returns original transcript and translated SRT files.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    src_lang = request.form.get('src_lang', 'en')
    tgt_lang = request.form.get('tgt_lang', 'es')
    
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_input.wav")
    
    # Convert to WAV if needed
    if audio_file.filename.lower().endswith('.wav'):
        audio_file.save(input_path)
    else:
        # Convert to WAV using ffmpeg
        temp_input = os.path.join(UPLOAD_DIR, f"{file_id}_temp_input")
        audio_file.save(temp_input)
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', temp_input, '-ar', '16000', '-ac', '1', input_path, '-y'
        ], check=True)
        os.unlink(temp_input)  # Clean up temp file
    
    try:
        # Step 1: Transcribe entire audio
        original_transcript = transcribe_audio_to_srt(input_path, src_lang)
        
        # Step 2: Translate transcript
        translated_transcript = translate_srt(original_transcript, src_lang, tgt_lang)
        
        # Return both files as a zip
        import zipfile
        zip_path = os.path.join(RESULTS_DIR, f"{file_id}_transcripts.zip")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(original_transcript, 'original_transcript.srt')
            zipf.write(translated_transcript, 'translated_transcript.srt')
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"{file_id}_transcripts.zip",
            mimetype='application/zip'
        )
        
    except Exception as e:
        print(f"[API] Error in transcribe-and-translate: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.unlink(input_path)

@app.route('/translate-audio-with-srt', methods=['POST'])
def translate_audio_with_srt_route():
    try:
        audio_file = request.files['audio']
        srt_file = request.files['srt']
        voice_clone = request.form.get('voice_clone', 'true').lower() == 'true'
        
        # Save uploaded files to temp
        input_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_input.wav")
        srt_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_input.srt")
        audio_file.save(input_path)
        srt_file.save(srt_path)
        
        print(f"[DEBUG] Loaded audio: {input_path}, srt: {srt_path}")
        
        output_path = process_audio_with_srt(input_path, srt_path, voice_clone)
        
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        import traceback
        print(f"[API] Error in translate-audio-with-srt: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def transcribe_audio_to_srt(audio_path, src_lang):
    """
    Transcribe audio to SRT format for human review.
    """
    import whisper
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Transcribe with timestamps
    result = model.transcribe(audio_path, language=src_lang, word_timestamps=True)
    
    # Convert to SRT format
    srt_content = ""
    for i, segment in enumerate(result['segments']):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        
        srt_content += f"{i+1}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"{text}\n\n"
    
    # Save SRT file
    srt_path = os.path.join(RESULTS_DIR, f"original_transcript_{os.path.basename(audio_path)}.srt")
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    return srt_path

def translate_srt(srt_path, src_lang, tgt_lang):
    """
    Translate SRT file from source to target language.
    """
    try:
        from transformers import pipeline
    except ImportError:
        print("[WARNING] transformers not available, using fallback translation")
        return srt_path  # Return original if transformers not available
    
    # Load translation pipeline
    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}")
    
    # Read SRT file
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse SRT and translate text
    lines = content.split('\n')
    translated_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.isdigit():  # Subtitle number
            translated_lines.append(line)
            i += 1
        elif '-->' in line:  # Timestamp
            translated_lines.append(line)
            i += 1
        elif line:  # Text to translate
            translated_text = translator(line)[0]['translation_text']
            translated_lines.append(translated_text)
            i += 1
        else:  # Empty line
            translated_lines.append(line)
            i += 1
    
    # Save translated SRT
    translated_content = '\n'.join(translated_lines)
    translated_path = os.path.join(RESULTS_DIR, f"translated_{os.path.basename(srt_path)}")
    with open(translated_path, 'w', encoding='utf-8') as f:
        f.write(translated_content)
    
    return translated_path

def process_audio_with_srt(audio_path, srt_path, voice_clone=True):
    """
    Process audio with voice cloning using reviewed SRT file.
    Uses chronological concatenation approach - no speed manipulation or time-stretching.
    """
    import librosa
    import soundfile as sf
    import tempfile
    import numpy as np
    import torch
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"[DEBUG] Loaded audio: {audio_path}, shape: {audio.shape}, sr: {sr}", flush=True)
    
    # Parse SRT file to get speech segments and translations
    speech_segments = parse_srt_to_segments(srt_path)
    print(f"[DEBUG] Parsed {len(speech_segments)} speech segments from SRT", flush=True)
    
    # Create temporary directory for segment files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Step 1: Run VAD to get non-speech segments
        try:
            from silero_vad import get_speech_timestamps
            # Load Silero VAD model using torch.hub (same as main.py)
            try:
                vad_model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
            except:
                vad_model = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
            
            # Convert numpy array to torch tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=int(sr))
            print(f"[DEBUG] VAD found {len(speech_timestamps)} speech segments", flush=True)
            
            # Convert speech timestamps to non-speech segments
            non_speech_segments = []
            last_end = 0.0
            
            for ts in speech_timestamps:
                start_time = ts['start'] / sr
                end_time = ts['end'] / sr
                
                # Add non-speech segment before this speech segment
                if start_time > last_end:
                    non_speech_segments.append({
                        'start': last_end,
                        'end': start_time,
                        'type': 'non_speech'
                    })
                
                last_end = end_time
            
            # Add final non-speech segment if needed
            if last_end < len(audio) / sr:
                non_speech_segments.append({
                    'start': last_end,
                    'end': len(audio) / sr,
                    'type': 'non_speech'
                })
            
            print(f"[DEBUG] Created {len(non_speech_segments)} non-speech segments", flush=True)
            
        except Exception as e:
            print(f"[WARNING] VAD failed: {e}, using fallback", flush=True)
            # Fallback: create one non-speech segment for the entire audio
            non_speech_segments = [{
                'start': 0.0,
                'end': len(audio) / sr,
                'type': 'non_speech'
            }]
        
        # Step 2: Generate TTS for each speech segment (natural speed, no manipulation)
        tts_segments = []
        for i, segment in enumerate(speech_segments):
            print(f"[DEBUG] Generating TTS for segment {i+1}/{len(speech_segments)}", flush=True)
            
            # Generate TTS at natural speed (no speed calculation)
            tts_audio = generate_tts(
                text=segment['text'],
                voice_clone=voice_clone,
                audio=audio,
                reference_start=segment['start'],
                reference_end=segment['end'],
                sr=int(sr)
            )
            
            # Save TTS segment to file
            tts_file = os.path.join(temp_dir, f"tts_segment_{i:04d}.wav")
            sf.write(tts_file, tts_audio, sr)
            
            tts_segments.append({
                'type': 'tts',
                'file': tts_file,
                'original_start': segment['start'],
                'original_end': segment['end'],
                'text': segment['text']
            })
            
            print(f"[DEBUG] Saved TTS segment {i+1} to {tts_file}", flush=True)
        
        # Step 3: Extract non-speech segments from original audio
        non_speech_files = []
        for i, segment in enumerate(non_speech_segments):
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            non_speech_audio = audio[start_sample:end_sample]
            
            # Save non-speech segment to file
            non_speech_file = os.path.join(temp_dir, f"non_speech_segment_{i:04d}.wav")
            sf.write(non_speech_file, non_speech_audio, sr)
            
            non_speech_files.append({
                'type': 'non_speech',
                'file': non_speech_file,
                'original_start': segment['start'],
                'original_end': segment['end']
            })
            
            print(f"[DEBUG] Saved non-speech segment {i+1} to {non_speech_file}", flush=True)
        
        # Step 4: Combine all segments in chronological order
        all_segments = tts_segments + non_speech_files
        all_segments.sort(key=lambda x: x['original_start'])
        
        print(f"[DEBUG] Combined {len(all_segments)} segments in chronological order", flush=True)
        
        # Step 5: Concatenate all segments
        final_audio_parts = []
        total_duration = 0.0
        
        for i, segment in enumerate(all_segments):
            # Load segment audio
            segment_audio, segment_sr = librosa.load(segment['file'], sr=sr)
            
            # Add to final audio
            final_audio_parts.append(segment_audio)
            
            segment_duration = len(segment_audio) / sr
            total_duration += segment_duration
            
            print(f"[DEBUG] Added {segment['type']} segment {i+1}: {segment_duration:.2f}s (total: {total_duration:.2f}s)", flush=True)
        
        # Concatenate all parts
        if final_audio_parts:
            final_audio = np.concatenate(final_audio_parts)
        else:
            final_audio = np.zeros(int(sr))  # 1 second of silence if no segments
        
        print(f"[DEBUG] Final audio duration: {len(final_audio)/sr:.2f}s", flush=True)
        
        # Step 6: Save final output
        output_path = os.path.join(RESULTS_DIR, f"final_output_{uuid.uuid4()}.wav")
        sf.write(output_path, final_audio, sr)
        
        print(f"[DEBUG] Final output saved to: {output_path}", flush=True)
        return output_path
        
    finally:
        # Cleanup temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def parse_srt_to_segments(srt_path):
    """
    Parse SRT file and extract speech segments with timing and text.
    """
    segments = []
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse SRT format
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # Parse timestamp line
            timestamp_line = lines[1]
            start_str, end_str = timestamp_line.split(' --> ')
            start_time = parse_timestamp(start_str)
            end_time = parse_timestamp(end_str)
            
            # Parse text (lines 2 onwards)
            text = ' '.join(lines[2:]).strip()
            
            segments.append({
                'start': start_time,
                'end': end_time,
                'text': text
            })
    
    return segments

def parse_timestamp(timestamp_str):
    """
    Parse SRT timestamp format (HH:MM:SS,mmm) to seconds.
    """
    time_part, ms_part = timestamp_str.split(',')
    h, m, s = map(int, time_part.split(':'))
    ms = int(ms_part)
    
    return h * 3600 + m * 60 + s + ms / 1000.0

def format_timestamp(seconds):
    """
    Format seconds to SRT timestamp format (HH:MM:SS,mmm).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def generate_tts(text, voice_clone, audio, reference_start=None, reference_end=None, sr=22050):
    """Generate TTS audio with optional voice cloning"""
    import librosa
    import numpy as np
    import tempfile
    import os
    import soundfile as sf
    
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    except ImportError:
        print("[WARNING] TTS not available, returning silence")
        return np.zeros(16000)  # 1 second of silence
    
    if voice_clone and reference_start is not None and reference_end is not None:
        # Extract reference audio segment for voice cloning
        start_sample = int(reference_start * sr)
        end_sample = int(reference_end * sr)
        reference_segment = audio[start_sample:end_sample]
        
        print(f"[DEBUG] Voice cloning: reference segment {reference_start:.2f}s to {reference_end:.2f}s, length: {len(reference_segment)} samples", flush=True)
        
        # Ensure we have enough audio for voice cloning (at least 1 second)
        min_samples = int(1.0 * sr)
        if len(reference_segment) < min_samples:
            # Pad with silence if needed
            padding = min_samples - len(reference_segment)
            reference_segment = np.concatenate([reference_segment, np.zeros(padding)])
            print(f"[DEBUG] Padded reference segment to {len(reference_segment)} samples", flush=True)
        
        # Save reference audio segment to temporary file for voice cloning
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_ref_file:
            # Ensure the audio is in the correct format for TTS
            reference_audio_resampled = librosa.resample(reference_segment, orig_sr=sr, target_sr=22050)
            # Convert to 16-bit PCM
            reference_audio_int16 = (reference_audio_resampled * 32767).astype(np.int16)
            # Save using soundfile for better compatibility
            sf.write(temp_ref_file.name, reference_audio_int16, 22050, 'PCM_16')
            temp_ref_file.flush()
            
            print(f"[DEBUG] Saved reference audio to {temp_ref_file.name}, duration: {len(reference_audio_resampled)/22050:.2f}s", flush=True)
            
            try:
                # Generate TTS with voice cloning using the reference audio file
                print(f"[DEBUG] Text length: {len(text)} chars, text: '{text[:50]}...'", flush=True)
                
                # Calculate speed to match original segment duration
                original_duration = reference_end - reference_start
                # Estimate TTS duration (roughly 15 characters per second for Spanish)
                estimated_tts_duration = len(text) / 15.0
                speed_factor = estimated_tts_duration / original_duration
                # Clamp speed to reasonable range
                speed_factor = max(0.5, min(2.0, speed_factor))
                print(f"[DEBUG] Original duration: {original_duration:.2f}s, estimated TTS: {estimated_tts_duration:.2f}s, speed: {speed_factor:.2f}", flush=True)
                
                # Try to use speed parameter - XTTS might support it
                try:
                    wav = tts.tts(text=text, language="es", speaker_wav=temp_ref_file.name, speed=speed_factor)
                    print(f"[DEBUG] TTS generated with speed={speed_factor}, output duration: {len(wav)/22050:.2f}s", flush=True)
                except (TypeError, ValueError) as e:
                    print(f"[DEBUG] Speed parameter not supported: {e}, using default speed", flush=True)
                    wav = tts.tts(text=text, language="es", speaker_wav=temp_ref_file.name)
                    print(f"[DEBUG] TTS generated with default speed, output duration: {len(wav)/22050:.2f}s", flush=True)
                
                # Ensure wav is a numpy array
                if not isinstance(wav, np.ndarray):
                    wav = np.array(wav)
                
                # Resample TTS output to match original audio sample rate
                if sr != 22050:
                    wav = librosa.resample(wav, orig_sr=22050, target_sr=sr)
                    print(f"[DEBUG] Resampled TTS from 22050 to {sr} Hz", flush=True)
                
                return wav
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_ref_file.name)
                except:
                    pass
    else:
        # Use basic TTS without voice cloning
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        wav = tts.tts(text=text, language="es", speaker="Claribel Dervla")
        return wav

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 