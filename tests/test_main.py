import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from io import StringIO
from video_translator.main import main


class TestMain(unittest.TestCase):
    
    def setUp(self):
        self.test_args = [
            'main.py',
            '--input', 'test_video.mp4',
            '--src-lang', 'en',
            '--tgt-lang', 'es',
            '--output', 'output_video.mp4'
        ]
    
    @patch('main.transcribe_video')
    @patch('main.translate_segments')
    @patch('main.generate_srt')
    @patch('main.burn_subtitles')
    def test_main_success(self, mock_burn, mock_generate, mock_translate, mock_transcribe):
        """Test successful main execution"""
        # Mock return values
        mock_segments = [{'start': 0, 'end': 5, 'text': 'Hello'}]
        mock_translated = [{'start': 0, 'end': 5, 'text': 'Hola'}]
        mock_srt_path = 'test_video_es.srt'
        
        mock_transcribe.return_value = mock_segments
        mock_translate.return_value = mock_translated
        mock_generate.return_value = mock_srt_path
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with patch.object(sys, 'argv', self.test_args):
                main()
        finally:
            sys.stdout = sys.__stdout__
        
        # Verify all functions were called correctly
        mock_transcribe.assert_called_once_with('test_video.mp4', 'en', debug=False)
        mock_translate.assert_called_once_with(mock_segments, 'en', 'es', debug=False)
        mock_generate.assert_called_once_with(mock_translated, 'test_video.mp4', 'es', debug=False)
        mock_burn.assert_called_once_with('test_video.mp4', mock_srt_path, 'output_video.mp4', debug=False)
        
        # Verify output contains success message
        output = captured_output.getvalue()
        self.assertIn("✅ Done!", output)
    
    @patch('main.transcribe_video')
    def test_main_file_not_found(self, mock_transcribe):
        """Test main with file not found error"""
        mock_transcribe.side_effect = FileNotFoundError("Video file not found: test_video.mp4")
        
        # Capture stdout and stderr
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with patch.object(sys, 'argv', self.test_args):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("❌ File not found", output)
    
    @patch('main.transcribe_video')
    def test_main_missing_dependency(self, mock_transcribe):
        """Test main with missing dependency error"""
        mock_transcribe.side_effect = ImportError("Whisper is not installed")
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with patch.object(sys, 'argv', self.test_args):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("❌ Missing dependency", output)
        self.assertIn("pip install -r requirements.txt", output)
    
    @patch('main.transcribe_video')
    def test_main_invalid_input(self, mock_transcribe):
        """Test main with invalid input error"""
        mock_transcribe.side_effect = ValueError("No segments provided")
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with patch.object(sys, 'argv', self.test_args):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("❌ Invalid input", output)
    
    @patch('main.transcribe_video')
    def test_main_processing_fails(self, mock_transcribe):
        """Test main with processing failure"""
        mock_transcribe.side_effect = RuntimeError("Transcription failed")
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with patch.object(sys, 'argv', self.test_args):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("❌ Processing failed", output)
    
    @patch('main.transcribe_video')
    def test_main_keyboard_interrupt(self, mock_transcribe):
        """Test main with keyboard interrupt"""
        mock_transcribe.side_effect = KeyboardInterrupt()
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with patch.object(sys, 'argv', self.test_args):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("❌ Process interrupted by user", output)
    
    @patch('main.transcribe_video')
    def test_main_debug_mode(self, mock_transcribe):
        """Test main with debug mode enabled"""
        # Add debug flag to args
        debug_args = self.test_args + ['--debug']
        
        # Mock return values
        mock_segments = [{'start': 0, 'end': 5, 'text': 'Hello'}]
        mock_transcribe.return_value = mock_segments
        
        # Mock other functions to avoid calling them
        with patch('main.translate_segments') as mock_translate, \
             patch('main.generate_srt') as mock_generate, \
             patch('main.burn_subtitles') as mock_burn:
            
            mock_translate.return_value = mock_segments
            mock_generate.return_value = 'test.srt'
            
            # Capture stdout
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                with patch.object(sys, 'argv', debug_args):
                    main()
            finally:
                sys.stdout = sys.__stdout__
            
            # Verify debug mode was passed to functions
            mock_transcribe.assert_called_once_with('test_video.mp4', 'en', debug=True)
            mock_translate.assert_called_once_with(mock_segments, 'en', 'es', debug=True)
            mock_generate.assert_called_once_with(mock_segments, 'test_video.mp4', 'es', debug=True)
            mock_burn.assert_called_once_with('test_video.mp4', 'test.srt', 'output_video.mp4', debug=True)


if __name__ == '__main__':
    unittest.main() 