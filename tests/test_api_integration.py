import os
import time
import tempfile
import pytest
from unittest.mock import patch
from video_translator.api import app, jobs

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('video_translator.api.process')
def test_full_pipeline(mock_process, client):
    # Mock process to create a dummy output file
    def fake_process(input_path, output_path, **kwargs):
        with open(output_path, 'wb') as f:
            f.write(b'\x00' * 2048)  # Write dummy video data
    mock_process.side_effect = fake_process

    test_mp4 = os.path.join(tempfile.gettempdir(), 'test_short.mp4')
    with open(test_mp4, 'wb') as f:
        f.write(b'\x00' * 1024)  # Dummy file, not used

    with open(test_mp4, 'rb') as f:
        data = {
            'video': (f, 'test_short.mp4'),
            'src_lang': 'en',
            'tgt_lang': 'es',
            'voice_clone': 'false',
            'audio_mode': 'subtitles-only',
            'original_volume': 0.3,
            'add_captions': 'false',
            'caption_font_size': 24,
        }
        resp = client.post('/process', data=data, content_type='multipart/form-data')
        assert resp.status_code == 200
        job_id = resp.get_json()['job_id']

    # Poll status
    for _ in range(10):  # Should be quick
        status = client.get(f'/status/{job_id}')
        assert status.status_code == 200
        data = status.get_json()
        if data['status'] == 'done':
            break
        elif data['status'] == 'error':
            pytest.fail(f"Pipeline failed: {data.get('error')}")
        time.sleep(0.2)
    else:
        pytest.fail("Pipeline did not complete in time")

    # Download result
    result = client.get(f'/result/{job_id}')
    assert result.status_code == 200
    assert len(result.data) > 1000  # Should be a non-empty file 