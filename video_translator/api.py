from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import threading
import uuid
import os
import tempfile
import traceback
from flask_cors import CORS

# Import your main pipeline logic (refactor main.py if needed)
from video_translator.main import process

app = Flask(__name__)
CORS(app)

# In-memory job store: {job_id: {status, progress, result_path, error}}
jobs = {}

UPLOAD_DIR = tempfile.gettempdir()
RESULTS_DIR = tempfile.gettempdir()


def run_pipeline(job_id, input_path, options):
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 0
        # Prepare output path with descriptive name
        orig_filename = jobs[job_id].get('orig_filename', 'video')
        base, _ = os.path.splitext(orig_filename)
        output_path = os.path.join(
            RESULTS_DIR,
            f"{base}_{options['src_lang']}_{options['tgt_lang']}_translated_{job_id}.mp4"
        )
        # Step-based progress updates
        def progress_hook(step):
            if step == 1:
                jobs[job_id]['progress'] = 20
            elif step == 2:
                jobs[job_id]['progress'] = 40
            elif step == 3:
                jobs[job_id]['progress'] = 60
            elif step == 4:
                jobs[job_id]['progress'] = 80
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
        jobs[job_id]['status'] = 'done'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['result_path'] = output_path
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['traceback'] = traceback.format_exc()

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
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'result_path': None,
        'error': None,
        'orig_filename': filename
    }
    thread = threading.Thread(target=run_pipeline, args=(job_id, input_path, options))
    thread.start()
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 