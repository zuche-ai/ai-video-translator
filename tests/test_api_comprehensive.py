import unittest
import tempfile
import os
import numpy as np
import librosa
import soundfile as sf
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import json

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_translator.api import (
    process_audio_with_srt,
    generate_tts,
    parse_srt,
    format_timestamp,
    create_segments_from_srt,
    splice_audio_with_segments
)


class TestVideoTranslationAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_audio = np.random.rand(22050 * 10)  # 10 seconds of random audio
        self.sample_sr = 22050
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Sample SRT content
        self.sample_srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
How are you today?

3
00:00:07,000 --> 00:00:09,000
I am doing well, thank you."""
        
        # Create test audio file
        self.test_audio_path = os.path.join(self.test_dir, "test_audio.wav")
        sf.write(self.test_audio_path, self.sample_audio, self.sample_sr)
        
        # Create test SRT file
        self.test_srt_path = os.path.join(self.test_dir, "test_srt.srt")
        with open(self.test_srt_path, 'w') as f:
            f.write(self.sample_srt_content)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_format_timestamp(self):
        """Test timestamp formatting"""
        # Test various time values
        self.assertEqual(format_timestamp(0), "00:00:00,000")
        self.assertEqual(format_timestamp(1.5), "00:00:01,500")
        self.assertEqual(format_timestamp(61.123), "00:01:01,123")
        self.assertEqual(format_timestamp(3661.999), "01:01:01,999")
    
    def test_parse_srt_valid(self):
        """Test parsing valid SRT content"""
        parsed = parse_srt(self.sample_srt_content)
        
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]['start'], 1.0)
        self.assertEqual(parsed[0]['end'], 3.0)
        self.assertEqual(parsed[0]['text'], 'Hello world')
        self.assertEqual(parsed[1]['text'], 'How are you today?')
        self.assertEqual(parsed[2]['text'], 'I am doing well, thank you.')
    
    def test_parse_srt_invalid_format(self):
        """Test parsing invalid SRT format"""
        invalid_srt = """1
00:00:01,000 --> 00:00:03,000
Hello world
2
invalid timestamp
How are you today?"""
        
        with self.assertRaises(ValueError):
            parse_srt(invalid_srt)
    
    def test_parse_srt_empty(self):
        """Test parsing empty SRT content"""
        parsed = parse_srt("")
        self.assertEqual(parsed, [])
    
    def test_create_segments_from_srt(self):
        """Test creating segments from SRT data"""
        srt_data = [
            {'start': 1.0, 'end': 3.0, 'text': 'Hello world'},
            {'start': 4.0, 'end': 6.0, 'text': 'How are you today?'}
        ]
        
        segments = create_segments_from_srt(srt_data)
        
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]['type'], 'tts')
        self.assertEqual(segments[0]['start'], 1.0)
        self.assertEqual(segments[0]['end'], 3.0)
        self.assertEqual(segments[0]['text'], 'Hello world')
    
    @patch('video_translator.api.TTS')
    def test_generate_tts_with_voice_cloning(self, mock_tts_class):
        """Test TTS generation with voice cloning"""
        # Mock TTS instance
        mock_tts = Mock()
        mock_tts.tts.return_value = np.random.rand(22050 * 2)  # 2 seconds of audio
        mock_tts_class.return_value = mock_tts
        
        # Test parameters
        text = "Hello world"
        voice_clone = True
        audio = np.random.rand(22050 * 10)  # 10 seconds of audio
        reference_start = 1.0
        reference_end = 3.0
        sr = 22050
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_ref.wav"
            mock_temp_file.return_value.__enter__.return_value.flush = Mock()
            
            result = generate_tts(text, voice_clone, audio, reference_start, reference_end, sr)
            
            # Verify TTS was called with correct parameters
            mock_tts.tts.assert_called_once()
            call_args = mock_tts.tts.call_args
            self.assertEqual(call_args[1]['text'], text)
            self.assertEqual(call_args[1]['language'], "es")
            self.assertEqual(call_args[1]['speaker_wav'], "/tmp/test_ref.wav")
            self.assertIn('speed', call_args[1])
    
    @patch('video_translator.api.TTS')
    def test_generate_tts_speed_calculation(self, mock_tts_class):
        """Test TTS speed calculation based on duration matching"""
        mock_tts = Mock()
        mock_tts.tts.return_value = np.random.rand(22050 * 2)
        mock_tts_class.return_value = mock_tts
        
        # Test with different text lengths and durations
        test_cases = [
            ("Short text", 1.0, 3.0, 0.5),  # Should be clamped to 0.5
            ("This is a medium length text that should take some time to speak", 1.0, 3.0, 1.0),  # Should be around 1.0
            ("This is a very long text that should take much longer to speak than the original audio duration", 1.0, 3.0, 2.0)  # Should be clamped to 2.0
        ]
        
        for text, start, end, expected_speed_range in test_cases:
            with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
                mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_ref.wav"
                mock_temp_file.return_value.__enter__.return_value.flush = Mock()
                
                result = generate_tts(text, True, np.random.rand(22050 * 10), start, end, 22050)
                
                call_args = mock_tts.tts.call_args
                actual_speed = call_args[1]['speed']
                self.assertGreaterEqual(actual_speed, 0.5)
                self.assertLessEqual(actual_speed, 2.0)
    
    @patch('video_translator.api.TTS')
    def test_generate_tts_without_voice_cloning(self, mock_tts_class):
        """Test TTS generation without voice cloning"""
        mock_tts = Mock()
        mock_tts.tts.return_value = np.random.rand(22050 * 2)
        mock_tts_class.return_value = mock_tts
        
        text = "Hello world"
        voice_clone = False
        audio = np.random.rand(22050 * 10)
        
        result = generate_tts(text, voice_clone, audio)
        
        # Should still call TTS but without speaker_wav
        mock_tts.tts.assert_called_once()
        call_args = mock_tts.tts.call_args
        self.assertEqual(call_args[1]['text'], text)
        self.assertEqual(call_args[1]['language'], "es")
        self.assertNotIn('speaker_wav', call_args[1])
    
    def test_splice_audio_with_segments(self):
        """Test audio splicing with TTS segments"""
        # Create test segments
        segments = [
            {
                'type': 'tts',
                'start': 1.0,
                'end': 3.0,
                'text': 'Hello world',
                'file': '/tmp/tts1.wav'
            },
            {
                'type': 'tts',
                'start': 4.0,
                'end': 6.0,
                'text': 'How are you?',
                'file': '/tmp/tts2.wav'
            }
        ]
        
        # Create mock TTS audio files
        tts1_audio = np.random.rand(22050 * 2)  # 2 seconds
        tts2_audio = np.random.rand(22050 * 2)  # 2 seconds
        
        with patch('librosa.load') as mock_load:
            mock_load.side_effect = [(tts1_audio, 22050), (tts2_audio, 22050)]
            
            # Create output audio buffer
            output_duration = 10  # seconds
            output_audio = np.zeros(int(output_duration * 22050))
            
            result = splice_audio_with_segments(segments, output_audio, 22050)
            
            # Verify librosa.load was called for each TTS segment
            self.assertEqual(mock_load.call_count, 2)
            
            # Verify output audio has content in the expected positions
            # (Note: This is a basic check - in practice you'd want more detailed verification)
            self.assertTrue(np.any(result != 0))
    
    def test_splice_audio_time_stretching(self):
        """Test audio time-stretching in splicing"""
        # Create a segment that needs time-stretching
        segments = [
            {
                'type': 'tts',
                'start': 1.0,
                'end': 3.0,  # 2 seconds original duration
                'text': 'Hello world',
                'file': '/tmp/tts1.wav'
            }
        ]
        
        # Create TTS audio that's 1 second (needs stretching to 2 seconds)
        tts_audio = np.random.rand(22050 * 1)  # 1 second
        
        with patch('librosa.load') as mock_load, \
             patch('librosa.effects.time_stretch') as mock_stretch:
            
            mock_load.return_value = (tts_audio, 22050)
            mock_stretch.return_value = np.random.rand(22050 * 2)  # Stretched to 2 seconds
            
            output_audio = np.zeros(int(10 * 22050))
            result = splice_audio_with_segments(segments, result, 22050)
            
            # Verify time-stretching was called
            mock_stretch.assert_called_once()
            call_args = mock_stretch.call_args
            self.assertEqual(call_args[1]['rate'], 2.0)  # Should stretch by factor of 2
    
    @patch('video_translator.api.generate_tts')
    @patch('video_translator.api.create_segments_from_srt')
    @patch('video_translator.api.parse_srt')
    @patch('librosa.load')
    @patch('soundfile.write')
    def test_process_audio_with_srt_full_pipeline(self, mock_sf_write, mock_librosa_load, 
                                                 mock_parse_srt, mock_create_segments, mock_generate_tts):
        """Test the full audio processing pipeline"""
        # Mock all the dependencies
        mock_librosa_load.return_value = (self.sample_audio, self.sample_sr)
        mock_sf_write.return_value = None
        
        srt_data = [
            {'start': 1.0, 'end': 3.0, 'text': 'Hello world'},
            {'start': 4.0, 'end': 6.0, 'text': 'How are you today?'}
        ]
        mock_parse_srt.return_value = srt_data
        
        segments = [
            {'type': 'tts', 'start': 1.0, 'end': 3.0, 'text': 'Hello world'},
            {'type': 'tts', 'start': 4.0, 'end': 6.0, 'text': 'How are you today?'}
        ]
        mock_create_segments.return_value = segments
        
        # Mock TTS generation
        tts_audio = np.random.rand(22050 * 2)
        mock_generate_tts.return_value = tts_audio
        
        # Test the full pipeline
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
            output_path = temp_output.name
        
        try:
            result_path = process_audio_with_srt(self.test_audio_path, self.test_srt_path, True)
            
            # Verify all the expected calls were made
            mock_librosa_load.assert_called_once_with(self.test_audio_path, sr=None)
            mock_parse_srt.assert_called_once()
            mock_create_segments.assert_called_once_with(srt_data)
            self.assertEqual(mock_generate_tts.call_count, 2)  # Called for each TTS segment
            mock_sf_write.assert_called()
            
            # Verify output file was created
            self.assertTrue(os.path.exists(result_path))
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_process_audio_with_srt_invalid_files(self):
        """Test processing with invalid input files"""
        # Test with non-existent audio file
        with self.assertRaises(FileNotFoundError):
            process_audio_with_srt("nonexistent_audio.wav", self.test_srt_path, True)
        
        # Test with non-existent SRT file
        with self.assertRaises(FileNotFoundError):
            process_audio_with_srt(self.test_audio_path, "nonexistent_srt.srt", True)
    
    def test_edge_cases(self):
        """Test various edge cases"""
        # Test with very short audio
        short_audio = np.random.rand(22050 * 1)  # 1 second
        short_audio_path = os.path.join(self.test_dir, "short_audio.wav")
        sf.write(short_audio_path, short_audio, self.sample_sr)
        
        # Test with empty SRT
        empty_srt_path = os.path.join(self.test_dir, "empty_srt.srt")
        with open(empty_srt_path, 'w') as f:
            f.write("")
        
        with patch('video_translator.api.parse_srt') as mock_parse_srt, \
             patch('video_translator.api.create_segments_from_srt') as mock_create_segments, \
             patch('video_translator.api.generate_tts') as mock_generate_tts, \
             patch('librosa.load') as mock_librosa_load, \
             patch('soundfile.write') as mock_sf_write:
            
            mock_librosa_load.return_value = (short_audio, self.sample_sr)
            mock_parse_srt.return_value = []
            mock_create_segments.return_value = []
            mock_sf_write.return_value = None
            
            result_path = process_audio_with_srt(short_audio_path, empty_srt_path, True)
            
            # Should handle empty SRT gracefully
            self.assertTrue(os.path.exists(result_path))
            mock_generate_tts.assert_not_called()  # No TTS segments to process
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large files"""
        # Create a larger audio file
        large_audio = np.random.rand(22050 * 60)  # 1 minute
        large_audio_path = os.path.join(self.test_dir, "large_audio.wav")
        sf.write(large_audio_path, large_audio, self.sample_sr)
        
        # Create SRT with many segments
        many_segments_srt = ""
        for i in range(100):
            start_time = i * 2
            end_time = start_time + 1.5
            many_segments_srt += f"{i+1}\n{format_timestamp(start_time)} --> {format_timestamp(end_time)}\nSegment {i+1}\n\n"
        
        many_segments_srt_path = os.path.join(self.test_dir, "many_segments.srt")
        with open(many_segments_srt_path, 'w') as f:
            f.write(many_segments_srt)
        
        with patch('video_translator.api.parse_srt') as mock_parse_srt, \
             patch('video_translator.api.create_segments_from_srt') as mock_create_segments, \
             patch('video_translator.api.generate_tts') as mock_generate_tts, \
             patch('librosa.load') as mock_librosa_load, \
             patch('soundfile.write') as mock_sf_write:
            
            mock_librosa_load.return_value = (large_audio, self.sample_sr)
            
            # Create many segments
            segments = []
            for i in range(100):
                segments.append({
                    'type': 'tts',
                    'start': i * 2,
                    'end': i * 2 + 1.5,
                    'text': f'Segment {i+1}'
                })
            
            mock_parse_srt.return_value = segments[:50]  # Parse returns first 50
            mock_create_segments.return_value = segments
            mock_generate_tts.return_value = np.random.rand(22050 * 1)
            mock_sf_write.return_value = None
            
            result_path = process_audio_with_srt(large_audio_path, many_segments_srt_path, True)
            
            # Should handle large files without memory issues
            self.assertTrue(os.path.exists(result_path))
            self.assertEqual(mock_generate_tts.call_count, 100)


if __name__ == '__main__':
    unittest.main() 