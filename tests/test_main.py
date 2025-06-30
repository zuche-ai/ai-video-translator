import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO
from video_translator.main import main

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_translator.core.transcriber import transcribe_video
from video_translator.core.translator import translate_segments
from video_translator.core.subtitles import generate_srt
from video_translator.video.video_editor import burn_subtitles
from video_translator.audio.voice_cloner import VoiceCloner


class TestMain(unittest.TestCase):
    
    def setUp(self):
        self.test_args = [
            'main.py',
            '--input', 'test_video.mp4',
            '--src-lang', 'en',
            '--tgt-lang', 'es',
            '--output', 'output_video.mp4'
        ]
    
    @patch('video_translator.main.transcribe_video')
    @patch('video_translator.main.translate_segments')
    @patch('video_translator.main.generate_srt')
    @patch('video_translator.main.burn_subtitles')
    def test_main_success(self, mock_burn, mock_generate, mock_translate, mock_transcribe):
        """Test successful main execution"""
        # Mock return values
        mock_segments = [
            {'start': 0.0, 'end': 2.0, 'text': 'Hello world'},
            {'start': 2.0, 'end': 4.0, 'text': 'How are you?'}
        ]
        mock_transcribe.return_value = mock_segments
        mock_translate.return_value = mock_segments
        mock_generate.return_value = 'test.srt'
        
        # Mock command line arguments
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        
        with patch('sys.argv', ['main.py'] + test_args):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.print') as mock_print:
                    main()
                    
                    # Verify the expected output
                    output = '\n'.join([call[0][0] if call[0] else '' for call in mock_print.call_args_list])
                    self.assertIn("✅ Done!", output)
    
    @patch('video_translator.main.transcribe_video')
    def test_main_file_not_found(self, mock_transcribe):
        """Test main with file not found error"""
        # Mock file not found
        mock_transcribe.side_effect = FileNotFoundError("Video file not found: test_video.mp4")
        
        # Mock command line arguments
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        
        with patch('sys.argv', ['main.py'] + test_args):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.print') as mock_print:
                    with self.assertRaises(SystemExit):
                        main()
                    
                    # Verify the expected output
                    output = '\n'.join([call[0][0] if call[0] else '' for call in mock_print.call_args_list])
                    self.assertIn("❌ File not found", output)
    
    @patch('video_translator.main.transcribe_video')
    def test_main_missing_dependency(self, mock_transcribe):
        """Test main with missing dependency error"""
        # Mock import error
        mock_transcribe.side_effect = ImportError("Missing dependency")
        
        # Mock command line arguments
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        
        with patch('sys.argv', ['main.py'] + test_args):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.print') as mock_print:
                    with self.assertRaises(SystemExit):
                        main()
                    
                    # Verify the expected output
                    output = '\n'.join([call[0][0] if call[0] else '' for call in mock_print.call_args_list])
                    self.assertIn("❌ Missing dependency", output)
    
    @patch('video_translator.main.transcribe_video')
    def test_main_invalid_input(self, mock_transcribe):
        """Test main with invalid input error"""
        # Mock value error
        mock_transcribe.side_effect = ValueError("Invalid input")
        
        # Mock command line arguments
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        
        with patch('sys.argv', ['main.py'] + test_args):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.print') as mock_print:
                    with self.assertRaises(SystemExit):
                        main()
                    
                    # Verify the expected output
                    output = '\n'.join([call[0][0] if call[0] else '' for call in mock_print.call_args_list])
                    self.assertIn("❌ Invalid input", output)
    
    @patch('video_translator.main.transcribe_video')
    def test_main_processing_fails(self, mock_transcribe):
        """Test main with processing failure"""
        # Mock runtime error
        mock_transcribe.side_effect = RuntimeError("Processing failed")
        
        # Mock command line arguments
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        
        with patch('sys.argv', ['main.py'] + test_args):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.print') as mock_print:
                    with self.assertRaises(SystemExit):
                        main()
                    
                    # Verify the expected output
                    output = '\n'.join([call[0][0] if call[0] else '' for call in mock_print.call_args_list])
                    self.assertIn("❌ Processing failed", output)
    
    @patch('video_translator.main.transcribe_video')
    def test_main_keyboard_interrupt(self, mock_transcribe):
        """Test main with keyboard interrupt"""
        # Mock keyboard interrupt
        mock_transcribe.side_effect = KeyboardInterrupt()
        
        # Mock command line arguments
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        
        with patch('sys.argv', ['main.py'] + test_args):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.print') as mock_print:
                    with self.assertRaises(SystemExit):
                        main()
                    
                    # Verify the expected output
                    output = '\n'.join([call[0][0] if call[0] else '' for call in mock_print.call_args_list])
                    self.assertIn("❌ Process interrupted by user", output)
    
    @patch('video_translator.main.transcribe_video')
    @patch('video_translator.main.translate_segments')
    @patch('video_translator.main.generate_srt')
    @patch('video_translator.main.burn_subtitles')
    def test_main_debug_mode(self, mock_burn, mock_generate, mock_translate, mock_transcribe):
        """Test main with debug mode enabled"""
        # Mock return values
        mock_segments = [
            {'start': 0.0, 'end': 2.0, 'text': 'Hello world'},
            {'start': 2.0, 'end': 4.0, 'text': 'How are you?'}
        ]
        mock_transcribe.return_value = mock_segments
        mock_translate.return_value = mock_segments
        mock_generate.return_value = 'test.srt'
        
        # Mock command line arguments with debug flag
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4', '--debug']
        
        with patch('sys.argv', ['main.py'] + test_args):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.print') as mock_print:
                    main()
                    
                    # Verify the expected output
                    output = '\n'.join([call[0][0] if call[0] else '' for call in mock_print.call_args_list])
                    self.assertIn("✅ Done!", output)


if __name__ == '__main__':
    unittest.main() 