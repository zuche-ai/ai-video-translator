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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 