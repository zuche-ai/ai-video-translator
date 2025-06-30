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
    
    @patch('video_translator.main.process')
    def test_main_success(self, mock_process):
        """Test successful main execution"""
        mock_process.return_value = None
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        with patch('sys.argv', ['main.py'] + test_args):
            main()  # Should not raise
    
    @patch('video_translator.main.process', side_effect=FileNotFoundError("Video file not found: test_video.mp4"))
    def test_main_file_not_found(self, mock_process):
        """Test main with file not found error"""
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        with patch('sys.argv', ['main.py'] + test_args):
            with self.assertRaises(SystemExit):
                main()
    
    @patch('video_translator.main.process', side_effect=ImportError("Missing dependency"))
    def test_main_missing_dependency(self, mock_process):
        """Test main with missing dependency error"""
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        with patch('sys.argv', ['main.py'] + test_args):
            with self.assertRaises(SystemExit):
                main()
    
    @patch('video_translator.main.process', side_effect=ValueError("Invalid input"))
    def test_main_invalid_input(self, mock_process):
        """Test main with invalid input error"""
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        with patch('sys.argv', ['main.py'] + test_args):
            with self.assertRaises(SystemExit):
                main()
    
    @patch('video_translator.main.process', side_effect=RuntimeError("Processing failed"))
    def test_main_processing_fails(self, mock_process):
        """Test main with processing failure"""
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        with patch('sys.argv', ['main.py'] + test_args):
            with self.assertRaises(SystemExit):
                main()
    
    @patch('video_translator.main.process', side_effect=KeyboardInterrupt())
    def test_main_keyboard_interrupt(self, mock_process):
        """Test main with keyboard interrupt"""
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4']
        with patch('sys.argv', ['main.py'] + test_args):
            with self.assertRaises(SystemExit):
                main()
    
    @patch('video_translator.main.process')
    def test_main_debug_mode(self, mock_process):
        """Test main with debug mode enabled"""
        mock_process.return_value = None
        test_args = ['--input', 'test_video.mp4', '--src-lang', 'en', '--tgt-lang', 'es', '--output', 'output.mp4', '--debug']
        with patch('sys.argv', ['main.py'] + test_args):
            main()  # Should not raise


if __name__ == '__main__':
    unittest.main() 