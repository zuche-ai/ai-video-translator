import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_translator.video.video_editor import burn_subtitles


class TestVideoEditor(unittest.TestCase):
    
    def setUp(self):
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.temp_dir, "test_video.mp4")
        self.test_srt_path = os.path.join(self.temp_dir, "test_subtitles.srt")
        self.test_output_path = os.path.join(self.temp_dir, "output_video.mp4")
        
        # Create test files
        with open(self.test_video_path, 'w') as f:
            f.write("fake video content")
        with open(self.test_srt_path, 'w') as f:
            f.write("fake subtitle content")
    
    def tearDown(self):
        # Clean up temporary files
        for file_path in [self.test_video_path, self.test_srt_path, self.test_output_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.rmdir(self.temp_dir)
    
    @patch('video_translator.video.video_editor.ffmpeg')
    def test_burn_subtitles_success(self, mock_ffmpeg):
        """Test successful subtitle burning"""
        # Mock ffmpeg components
        mock_input = Mock()
        mock_video = Mock()
        mock_audio = Mock()
        mock_output = Mock()
        mock_stream = Mock()
        
        mock_ffmpeg.input.return_value = mock_input
        mock_input.video.filter.return_value = mock_video
        mock_input.audio = mock_audio
        mock_ffmpeg.output.return_value = mock_stream
        
        # Test the function
        burn_subtitles(self.test_video_path, self.test_srt_path, self.test_output_path, debug=False)
        
        # Verify ffmpeg calls
        mock_ffmpeg.input.assert_called_once_with(self.test_video_path)
        mock_input.video.filter.assert_called_once_with('subtitles', self.test_srt_path)
        mock_ffmpeg.output.assert_called_once_with(mock_video, mock_audio, self.test_output_path, vcodec='libx264', acodec='aac')
        mock_ffmpeg.run.assert_called_once_with(mock_stream, overwrite_output=True, quiet=True)
    
    def test_burn_subtitles_video_file_not_found(self):
        """Test error when video file doesn't exist"""
        with self.assertRaises(FileNotFoundError) as context:
            burn_subtitles("nonexistent_video.mp4", self.test_srt_path, self.test_output_path)
        
        self.assertIn("Video file not found", str(context.exception))
    
    def test_burn_subtitles_srt_file_not_found(self):
        """Test error when SRT file doesn't exist"""
        with self.assertRaises(FileNotFoundError) as context:
            burn_subtitles(self.test_video_path, "nonexistent_subtitles.srt", self.test_output_path)
        
        self.assertIn("SRT file not found", str(context.exception))
    
    @patch('video_translator.video.video_editor.ffmpeg')
    def test_burn_subtitles_ffmpeg_not_installed(self, mock_ffmpeg):
        """Test error when ffmpeg-python is not installed"""
        mock_ffmpeg.input.side_effect = ImportError("No module named 'ffmpeg'")
        
        with self.assertRaises(ImportError) as context:
            burn_subtitles(self.test_video_path, self.test_srt_path, self.test_output_path)
        
        self.assertIn("Ffmpeg-python is not installed", str(context.exception))
    
    @patch('video_translator.video.video_editor.ffmpeg')
    def test_burn_subtitles_processing_fails(self, mock_ffmpeg):
        """Test error when video processing fails"""
        # Mock ffmpeg components
        mock_input = Mock()
        mock_video = Mock()
        mock_audio = Mock()
        mock_output = Mock()
        mock_stream = Mock()
        
        mock_ffmpeg.input.return_value = mock_input
        mock_input.video.filter.return_value = mock_video
        mock_input.audio = mock_audio
        mock_ffmpeg.output.return_value = mock_stream
        mock_ffmpeg.run.side_effect = Exception("FFmpeg processing error")
        
        with self.assertRaises(RuntimeError) as context:
            burn_subtitles(self.test_video_path, self.test_srt_path, self.test_output_path)
        
        self.assertIn("Video processing failed", str(context.exception))
    
    @patch('video_translator.video.video_editor.ffmpeg')
    def test_burn_subtitles_debug_mode(self, mock_ffmpeg):
        """Test subtitle burning with debug mode enabled"""
        # Mock ffmpeg components
        mock_input = Mock()
        mock_video = Mock()
        mock_audio = Mock()
        mock_output = Mock()
        mock_stream = Mock()
        
        mock_ffmpeg.input.return_value = mock_input
        mock_input.video.filter.return_value = mock_video
        mock_input.audio = mock_audio
        mock_ffmpeg.output.return_value = mock_stream
        
        # Test with debug=True
        burn_subtitles(self.test_video_path, self.test_srt_path, self.test_output_path, debug=True)
        
        # Verify ffmpeg.run was called with quiet=False for debug mode
        mock_ffmpeg.run.assert_called_once_with(mock_stream, overwrite_output=True, quiet=False)


if __name__ == '__main__':
    unittest.main() 