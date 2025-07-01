import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_translator.core.subtitles import generate_srt


class TestSubtitles(unittest.TestCase):
    
    def setUp(self):
        self.test_segments = [
            {'start': 0.0, 'end': 5.0, 'text': 'Hello world'},
            {'start': 5.0, 'end': 10.0, 'text': 'How are you?'}
        ]
        self.test_video_path = "test_video.mp4"
        self.test_language = "es"
    
    def test_generate_srt_success(self):
        """Test successful SRT file generation"""
        with patch('video_translator.core.subtitles.pysrt') as mock_pysrt:
            # Mock pysrt components
            mock_subs = Mock()
            mock_pysrt.SubRipFile.return_value = mock_subs
            
            # Test the function
            result = generate_srt(self.test_segments, self.test_video_path, self.test_language, debug=False)
            
            # Verify results
            expected_filename = "test_video_es.srt"
            self.assertEqual(result, expected_filename)
            self.assertEqual(mock_subs.append.call_count, 2)
            mock_subs.save.assert_called_once_with(expected_filename, encoding='utf-8')
    
    def test_generate_srt_empty_list(self):
        """Test error when segments list is empty"""
        with self.assertRaises(ValueError) as context:
            generate_srt([], self.test_video_path, self.test_language)
        
        self.assertIn("No segments provided", str(context.exception))
    
    def test_generate_srt_missing_start_field(self):
        """Test error when segment is missing start field"""
        invalid_segments = [
            {'end': 5.0, 'text': 'Hello'},  # Missing start field
            {'start': 5.0, 'end': 10.0, 'text': 'How are you?'}
        ]
        
        with self.assertRaises(ValueError) as context:
            generate_srt(invalid_segments, self.test_video_path, self.test_language)
        
        self.assertIn("missing required fields", str(context.exception))
    
    def test_generate_srt_missing_end_field(self):
        """Test error when segment is missing end field"""
        invalid_segments = [
            {'start': 0.0, 'text': 'Hello'},  # Missing end field
            {'start': 5.0, 'end': 10.0, 'text': 'How are you?'}
        ]
        
        with self.assertRaises(ValueError) as context:
            generate_srt(invalid_segments, self.test_video_path, self.test_language)
        
        self.assertIn("missing required fields", str(context.exception))
    
    def test_generate_srt_missing_text_field(self):
        """Test error when segment is missing text field"""
        invalid_segments = [
            {'start': 0.0, 'end': 5.0},  # Missing text field
            {'start': 5.0, 'end': 10.0, 'text': 'How are you?'}
        ]
        
        with self.assertRaises(ValueError) as context:
            generate_srt(invalid_segments, self.test_video_path, self.test_language)
        
        self.assertIn("missing required fields", str(context.exception))
    
    def test_generate_srt_save_fails(self):
        """Test error when SRT file save fails"""
        with patch('video_translator.core.subtitles.pysrt') as mock_pysrt:
            mock_subs = Mock()
            mock_subs.save.side_effect = Exception("Save error")
            mock_pysrt.SubRipFile.return_value = mock_subs
            
            with self.assertRaises(RuntimeError) as context:
                generate_srt(self.test_segments, self.test_video_path, self.test_language)
            
            self.assertIn("SRT file creation failed", str(context.exception))
    
    def test_generate_srt_debug_mode(self):
        """Test SRT generation with debug mode enabled"""
        with patch('video_translator.core.subtitles.pysrt') as mock_pysrt:
            mock_subs = Mock()
            mock_pysrt.SubRipFile.return_value = mock_subs
            
            # Test with debug=True (should not raise any errors)
            result = generate_srt(self.test_segments, self.test_video_path, self.test_language, debug=True)
            self.assertEqual(result, "test_video_es.srt")
    
    def test_generate_srt_filename_generation(self):
        """Test SRT filename generation with different video paths"""
        test_cases = [
            ("video.mp4", "es", "video_es.srt"),
            ("my_video.avi", "fr", "my_video_fr.srt"),
            ("path/to/video.mov", "de", "video_de.srt"),
            ("video_with_spaces.mp4", "it", "video_with_spaces_it.srt")
        ]
        
        with patch('video_translator.core.subtitles.pysrt') as mock_pysrt:
            mock_subs = Mock()
            mock_pysrt.SubRipFile.return_value = mock_subs
            
            for video_path, language, expected_filename in test_cases:
                result = generate_srt(self.test_segments, video_path, language, debug=False)
                self.assertEqual(result, expected_filename)


if __name__ == '__main__':
    unittest.main() 