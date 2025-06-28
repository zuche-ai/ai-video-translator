from video_translator.transcriber import transcribe_video 

import unittest
from unittest.mock import Mock, patch, MagicMock
import warnings
import tempfile
import os


class TestTranscriber(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary video file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.temp_dir, "test_video.mp4")
        with open(self.test_video_path, 'w') as f:
            f.write("fake video content")
    
    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
        os.rmdir(self.temp_dir)
    
    @patch('video_translator.transcriber.whisper')
    @patch('video_translator.transcriber.torch')
    def test_transcribe_video_suppresses_fp16_warning_on_cpu(self, mock_torch, mock_whisper):
        """Test that FP16 warning is suppressed on CPU"""
        # Mock torch to simulate CPU environment
        mock_torch.cuda.is_available.return_value = False
        
        # Mock whisper model and transcribe method
        mock_model = Mock()
        mock_result = {
            'segments': [
                {'start': 0.0, 'end': 5.0, 'text': 'Hello world'}
            ]
        }
        mock_model.transcribe.return_value = mock_result
        mock_whisper.load_model.return_value = mock_model
        
        # Capture warnings to verify they're suppressed
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test the function
            result = transcribe_video(self.test_video_path, 'en', debug=False)
            
            # Verify no FP16 warnings were raised
            fp16_warnings = [warning for warning in w if "FP16 is not supported on CPU" in str(warning.message)]
            self.assertEqual(len(fp16_warnings), 0, "FP16 warning should be suppressed on CPU")
        
        # Verify results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'Hello world')
    
    @patch('video_translator.transcriber.whisper')
    @patch('video_translator.transcriber.torch')
    def test_transcribe_video_success(self, mock_torch, mock_whisper):
        """Test successful transcription"""
        # Mock torch to simulate GPU environment
        mock_torch.cuda.is_available.return_value = True
        
        # Mock whisper model and transcribe method
        mock_model = Mock()
        mock_result = {
            'segments': [
                {'start': 0.0, 'end': 5.0, 'text': 'Hello world'},
                {'start': 5.0, 'end': 10.0, 'text': 'How are you?'}
            ]
        }
        mock_model.transcribe.return_value = mock_result
        mock_whisper.load_model.return_value = mock_model
        
        # Test the function
        result = transcribe_video(self.test_video_path, 'en', debug=False)
        
        # Verify results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], 'Hello world')
        self.assertEqual(result[1]['text'], 'How are you?')
        mock_whisper.load_model.assert_called_once_with("base")
        mock_model.transcribe.assert_called_once_with(self.test_video_path, language='en')
    
    def test_transcribe_video_file_not_found(self):
        """Test error when video file doesn't exist"""
        with self.assertRaises(FileNotFoundError) as context:
            transcribe_video("nonexistent_video.mp4", 'en')
        
        self.assertIn("Video file not found", str(context.exception))
    
    @patch('video_translator.transcriber.whisper')
    def test_transcribe_video_whisper_not_installed(self, mock_whisper):
        """Test error when whisper is not installed"""
        mock_whisper.load_model.side_effect = ImportError("No module named 'whisper'")
        
        with self.assertRaises(ImportError) as context:
            transcribe_video(self.test_video_path, 'en')
        
        self.assertIn("Whisper is not installed", str(context.exception))
    
    @patch('video_translator.transcriber.whisper')
    def test_transcribe_video_transcription_fails(self, mock_whisper):
        """Test error when transcription fails"""
        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Transcription error")
        mock_whisper.load_model.return_value = mock_model
        
        with self.assertRaises(RuntimeError) as context:
            transcribe_video(self.test_video_path, 'en')
        
        self.assertIn("Transcription failed", str(context.exception))
    
    @patch('video_translator.transcriber.whisper')
    def test_transcribe_video_debug_mode(self, mock_whisper):
        """Test transcription with debug mode enabled"""
        mock_model = Mock()
        mock_result = {'segments': []}
        mock_model.transcribe.return_value = mock_result
        mock_whisper.load_model.return_value = mock_model
        
        # Test with debug=True (should not raise any errors)
        result = transcribe_video(self.test_video_path, 'en', debug=True)
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main() 