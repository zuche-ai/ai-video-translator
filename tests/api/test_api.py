import os
import tempfile
import pytest
from video_translator.api import app, jobs

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_process_endpoint(client):
    # Create a small dummy video file
    dummy_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    dummy_video.write(b'\x00' * 1024)  # Not a real video, just for API test
    dummy_video.seek(0)
    dummy_video.close()

    data = {
        'video': (open(dummy_video.name, 'rb'), 'test.mp4'),
        'src_lang': 'en',
        'tgt_lang': 'es',
        'voice_clone': 'false',
        'audio_mode': 'subtitles-only',
        'original_volume': 0.3,
        'add_captions': 'false',
        'caption_font_size': 24,
    }
    response = client.post('/process', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    job_id = response.get_json()['job_id']
    assert job_id in jobs

    # Clean up
    os.unlink(dummy_video.name) 