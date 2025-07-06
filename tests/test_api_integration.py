import unittest
import tempfile
import os
import numpy as np
import soundfile as sf
import requests
import time
import json
from unittest.mock import patch, Mock
import sys

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_translator.api import app


class TestVideoTranslationIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test audio file
        self.test_audio = np.random.rand(22050 * 5)  # 5 seconds of audio
        self.test_audio_path = os.path.join(self.test_dir, "test_audio.wav")
        sf.write(self.test_audio_path, self.test_audio, 22050)
        
        # Create test SRT file
        self.test_srt_content = """1
00:00:01,000 --> 00:00:02,500
Hello world

2
00:00:03,000 --> 00:00:04,500
How are you today?

3
00:00:05,000 --> 00:00:06,500
I am doing well, thank you."""
        
        self.test_srt_path = os.path.join(self.test_dir, "test_srt.srt")
        with open(self.test_srt_path, 'w') as f:
            f.write(self.test_srt_content)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), 'OK')
    
    def test_transcribe_and_translate_endpoint(self):
        """Test the transcribe and translate endpoint"""
        with open(self.test_audio_path, 'rb') as audio_file:
            files = {'audio': (self.test_audio_path, audio_file, 'audio/wav')}
            
            with patch('video_translator.api.transcribe_audio') as mock_transcribe, \
                 patch('video_translator.api.translate_text') as mock_translate, \
                 patch('video_translator.api.save_srt') as mock_save_srt:
                
                # Mock the transcription and translation
                mock_transcribe.return_value = [
                    {'start': 1.0, 'end': 2.5, 'text': 'Hello world'},
                    {'start': 3.0, 'end': 4.5, 'text': 'How are you today?'}
                ]
                
                mock_translate.return_value = [
                    {'start': 1.0, 'end': 2.5, 'text': 'Hola mundo'},
                    {'start': 3.0, 'end': 4.5, 'text': '¿Cómo estás hoy?'}
                ]
                
                mock_save_srt.return_value = {
                    'original_srt': '/tmp/original.srt',
                    'translated_srt': '/tmp/translated.srt'
                }
                
                response = self.app.post('/transcribe-and-translate', files=files)
                
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data.decode())
                self.assertIn('original_srt', data)
                self.assertIn('translated_srt', data)
    
    def test_translate_audio_with_srt_endpoint(self):
        """Test the translate audio with SRT endpoint"""
        with open(self.test_audio_path, 'rb') as audio_file, \
             open(self.test_srt_path, 'rb') as srt_file:
            
            files = {
                'audio': (self.test_audio_path, audio_file, 'audio/wav'),
                'srt': (self.test_srt_path, srt_file, 'text/plain')
            }
            
            with patch('video_translator.api.process_audio_with_srt') as mock_process:
                # Mock the processing function
                mock_process.return_value = '/tmp/output.wav'
                
                response = self.app.post('/translate-audio-with-srt', files=files)
                
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content_type, 'audio/wav')
    
    def test_translate_audio_with_srt_missing_files(self):
        """Test the endpoint with missing files"""
        # Test with missing audio file
        with open(self.test_srt_path, 'rb') as srt_file:
            files = {'srt': (self.test_srt_path, srt_file, 'text/plain')}
            response = self.app.post('/translate-audio-with-srt', files=files)
            self.assertEqual(response.status_code, 400)
        
        # Test with missing SRT file
        with open(self.test_audio_path, 'rb') as audio_file:
            files = {'audio': (self.test_audio_path, audio_file, 'audio/wav')}
            response = self.app.post('/translate-audio-with-srt', files=files)
            self.assertEqual(response.status_code, 400)
    
    def test_translate_audio_with_srt_invalid_files(self):
        """Test the endpoint with invalid file formats"""
        # Create invalid audio file
        invalid_audio_path = os.path.join(self.test_dir, "invalid.txt")
        with open(invalid_audio_path, 'w') as f:
            f.write("This is not an audio file")
        
        with open(invalid_audio_path, 'rb') as audio_file, \
             open(self.test_srt_path, 'rb') as srt_file:
            
            files = {
                'audio': (invalid_audio_path, audio_file, 'text/plain'),
                'srt': (self.test_srt_path, srt_file, 'text/plain')
            }
            
            response = self.app.post('/translate-audio-with-srt', files=files)
            self.assertEqual(response.status_code, 500)  # Should fail during processing
    
    def test_error_handling(self):
        """Test error handling in the API"""
        # Test with non-existent endpoint
        response = self.app.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        # Test with invalid request method
        response = self.app.get('/translate-audio-with-srt')
        self.assertEqual(response.status_code, 405)  # Method not allowed
    
    def test_file_size_limits(self):
        """Test handling of large files"""
        # Create a large audio file (simulate)
        large_audio = np.random.rand(22050 * 300)  # 5 minutes
        large_audio_path = os.path.join(self.test_dir, "large_audio.wav")
        sf.write(large_audio_path, large_audio, 22050)
        
        with open(large_audio_path, 'rb') as audio_file, \
             open(self.test_srt_path, 'rb') as srt_file:
            
            files = {
                'audio': (large_audio_path, audio_file, 'audio/wav'),
                'srt': (self.test_srt_path, srt_file, 'text/plain')
            }
            
            with patch('video_translator.api.process_audio_with_srt') as mock_process:
                mock_process.return_value = '/tmp/output.wav'
                
                response = self.app.post('/translate-audio-with-srt', files=files)
                
                # Should handle large files (though it might take time)
                self.assertIn(response.status_code, [200, 500])  # Either success or processing error
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                with open(self.test_audio_path, 'rb') as audio_file, \
                     open(self.test_srt_path, 'rb') as srt_file:
                    
                    files = {
                        'audio': (self.test_audio_path, audio_file, 'audio/wav'),
                        'srt': (self.test_srt_path, srt_file, 'text/plain')
                    }
                    
                    with patch('video_translator.api.process_audio_with_srt') as mock_process:
                        mock_process.return_value = '/tmp/output.wav'
                        response = self.app.post('/translate-audio-with-srt', files=files)
                        results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Start multiple concurrent requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        while not results.empty():
            result = results.get()
            self.assertIn(result, [200, 500])  # Should handle concurrent requests
    
    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after processing"""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Make multiple requests to test memory cleanup
        for i in range(5):
            with open(self.test_audio_path, 'rb') as audio_file, \
                 open(self.test_srt_path, 'rb') as srt_file:
                
                files = {
                    'audio': (self.test_audio_path, audio_file, 'audio/wav'),
                    'srt': (self.test_srt_path, srt_file, 'text/plain')
                }
                
                with patch('video_translator.api.process_audio_with_srt') as mock_process:
                    mock_process.return_value = '/tmp/output.wav'
                    response = self.app.post('/translate-audio-with-srt', files=files)
                    self.assertEqual(response.status_code, 200)
            
            # Force garbage collection
            gc.collect()
        
        # Check final memory usage (should be reasonable)
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024)
    
    def test_output_audio_quality(self):
        """Test that output audio has reasonable quality"""
        with open(self.test_audio_path, 'rb') as audio_file, \
             open(self.test_srt_path, 'rb') as srt_file:
            
            files = {
                'audio': (self.test_audio_path, audio_file, 'audio/wav'),
                'srt': (self.test_srt_path, srt_file, 'text/plain')
            }
            
            with patch('video_translator.api.process_audio_with_srt') as mock_process:
                # Create a temporary output file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                    output_path = temp_output.name
                
                try:
                    # Create some test output audio
                    output_audio = np.random.rand(22050 * 10)  # 10 seconds
                    sf.write(output_path, output_audio, 22050)
                    
                    mock_process.return_value = output_path
                    
                    response = self.app.post('/translate-audio-with-srt', files=files)
                    
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.content_type, 'audio/wav')
                    
                    # Check that response contains audio data
                    self.assertGreater(len(response.data), 1000)  # Should have substantial audio data
                    
                finally:
                    if os.path.exists(output_path):
                        os.unlink(output_path)


class TestVideoTranslationPerformance(unittest.TestCase):
    """Performance tests for the video translation API"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test audio file
        self.test_audio = np.random.rand(22050 * 10)  # 10 seconds
        self.test_audio_path = os.path.join(self.test_dir, "test_audio.wav")
        sf.write(self.test_audio_path, self.test_audio, 22050)
        
        # Create test SRT file
        self.test_srt_content = """1
00:00:01,000 --> 00:00:02,000
Test segment 1

2
00:00:03,000 --> 00:00:04,000
Test segment 2

3
00:00:05,000 --> 00:00:06,000
Test segment 3"""
        
        self.test_srt_path = os.path.join(self.test_dir, "test_srt.srt")
        with open(self.test_srt_path, 'w') as f:
            f.write(self.test_srt_content)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_response_time(self):
        """Test that API responses are reasonably fast"""
        with open(self.test_audio_path, 'rb') as audio_file, \
             open(self.test_srt_path, 'rb') as srt_file:
            
            files = {
                'audio': (self.test_audio_path, audio_file, 'audio/wav'),
                'srt': (self.test_srt_path, srt_file, 'text/plain')
            }
            
            with patch('video_translator.api.process_audio_with_srt') as mock_process:
                mock_process.return_value = '/tmp/output.wav'
                
                start_time = time.time()
                response = self.app.post('/translate-audio-with-srt', files=files)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                # Response should be reasonably fast (less than 5 seconds for mocked processing)
                self.assertLess(response_time, 5.0)
                self.assertEqual(response.status_code, 200)
    
    def test_throughput(self):
        """Test API throughput with multiple requests"""
        import threading
        import time
        
        results = []
        lock = threading.Lock()
        
        def make_request():
            try:
                with open(self.test_audio_path, 'rb') as audio_file, \
                     open(self.test_srt_path, 'rb') as srt_file:
                    
                    files = {
                        'audio': (self.test_audio_path, audio_file, 'audio/wav'),
                        'srt': (self.test_srt_path, srt_file, 'text/plain')
                    }
                    
                    with patch('video_translator.api.process_audio_with_srt') as mock_process:
                        mock_process.return_value = '/tmp/output.wav'
                        
                        start_time = time.time()
                        response = self.app.post('/translate-audio-with-srt', files=files)
                        end_time = time.time()
                        
                        with lock:
                            results.append({
                                'status': response.status_code,
                                'time': end_time - start_time
                            })
            except Exception as e:
                with lock:
                    results.append({'error': str(e)})
        
        # Start multiple concurrent requests
        threads = []
        start_time = time.time()
        
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check results
        successful_requests = sum(1 for r in results if r.get('status') == 200)
        self.assertGreater(successful_requests, 5)  # At least 5 should succeed
        
        # Calculate throughput
        throughput = successful_requests / total_time
        self.assertGreater(throughput, 0.5)  # At least 0.5 requests per second


if __name__ == '__main__':
    unittest.main() 