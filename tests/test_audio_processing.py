import unittest
import tempfile
import os
import numpy as np
import librosa
import soundfile as sf
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_translator.api import (
    generate_tts,
    splice_audio_with_segments,
    create_segments_from_srt,
    parse_srt
)


class TestAudioProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.sample_sr = 22050
        
        # Create sample audio data
        self.sample_audio = np.random.rand(self.sample_sr * 10)  # 10 seconds
        
        # Create sample SRT content
        self.sample_srt = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
How are you today?

3
00:00:07,000 --> 00:00:09,000
I am doing well, thank you."""
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    @patch('video_translator.api.TTS')
    def test_generate_tts_voice_cloning_parameters(self, mock_tts_class):
        """Test TTS generation with voice cloning parameters"""
        mock_tts = Mock()
        mock_tts.tts.return_value = np.random.rand(22050 * 2)
        mock_tts_class.return_value = mock_tts
        
        text = "Test text for TTS generation"
        voice_clone = True
        audio = np.random.rand(22050 * 10)
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
            
            # Check required parameters
            self.assertEqual(call_args[1]['text'], text)
            self.assertEqual(call_args[1]['language'], "es")
            self.assertEqual(call_args[1]['speaker_wav'], "/tmp/test_ref.wav")
            
            # Check speed parameter exists and is reasonable
            self.assertIn('speed', call_args[1])
            speed = call_args[1]['speed']
            self.assertGreaterEqual(speed, 0.5)
            self.assertLessEqual(speed, 2.0)
    
    @patch('video_translator.api.TTS')
    def test_generate_tts_speed_calculation_accuracy(self, mock_tts_class):
        """Test TTS speed calculation accuracy"""
        mock_tts = Mock()
        mock_tts.tts.return_value = np.random.rand(22050 * 2)
        mock_tts_class.return_value = mock_tts
        
        # Test cases: (text, original_duration, expected_speed_range)
        test_cases = [
            ("Short", 1.0, (0.5, 1.0)),  # Short text, long duration -> slow speed
            ("This is a medium length text that should take some time", 2.0, (0.8, 1.5)),  # Medium
            ("This is a very long text that should take much longer to speak than the original audio duration", 1.0, (1.5, 2.0))  # Long text, short duration -> fast speed
        ]
        
        for text, original_duration, expected_range in test_cases:
            with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
                mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_ref.wav"
                mock_temp_file.return_value.__enter__.return_value.flush = Mock()
                
                result = generate_tts(text, True, np.random.rand(22050 * 10), 0, original_duration, 22050)
                
                call_args = mock_tts.tts.call_args
                actual_speed = call_args[1]['speed']
                
                self.assertGreaterEqual(actual_speed, expected_range[0])
                self.assertLessEqual(actual_speed, expected_range[1])
    
    @patch('video_translator.api.TTS')
    def test_generate_tts_reference_audio_format(self, mock_tts_class):
        """Test that reference audio is properly formatted for TTS"""
        mock_tts = Mock()
        mock_tts.tts.return_value = np.random.rand(22050 * 2)
        mock_tts_class.return_value = mock_tts
        
        text = "Test text"
        voice_clone = True
        audio = np.random.rand(22050 * 10)
        reference_start = 1.0
        reference_end = 3.0
        sr = 22050
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
             patch('librosa.resample') as mock_resample, \
             patch('soundfile.write') as mock_sf_write:
            
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_ref.wav"
            mock_temp_file.return_value.__enter__.return_value.flush = Mock()
            mock_resample.return_value = np.random.rand(22050 * 2)
            mock_sf_write.return_value = None
            
            result = generate_tts(text, voice_clone, audio, reference_start, reference_end, sr)
            
            # Verify resampling was called with correct parameters
            mock_resample.assert_called_once()
            resample_args = mock_resample.call_args
            self.assertEqual(resample_args[1]['orig_sr'], sr)
            self.assertEqual(resample_args[1]['target_sr'], 22050)
            
            # Verify soundfile write was called with correct parameters
            mock_sf_write.assert_called_once()
            sf_write_args = mock_sf_write.call_args
            self.assertEqual(sf_write_args[1]['samplerate'], 22050)
            self.assertEqual(sf_write_args[1]['subtype'], 'PCM_16')
    
    @patch('video_translator.api.TTS')
    def test_generate_tts_short_reference_audio(self, mock_tts_class):
        """Test TTS with very short reference audio (should pad)"""
        mock_tts = Mock()
        mock_tts.tts.return_value = np.random.rand(22050 * 2)
        mock_tts_class.return_value = mock_tts
        
        text = "Test text"
        voice_clone = True
        audio = np.random.rand(22050 * 10)
        reference_start = 1.0
        reference_end = 1.1  # Very short reference (0.1 seconds)
        sr = 22050
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
             patch('librosa.resample') as mock_resample, \
             patch('soundfile.write') as mock_sf_write:
            
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_ref.wav"
            mock_temp_file.return_value.__enter__.return_value.flush = Mock()
            mock_resample.return_value = np.random.rand(22050 * 2)
            mock_sf_write.return_value = None
            
            result = generate_tts(text, voice_clone, audio, reference_start, reference_end, sr)
            
            # Should handle short reference audio gracefully
            mock_tts.tts.assert_called_once()
    
    def test_splice_audio_with_segments_basic(self):
        """Test basic audio splicing functionality"""
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
        
        # Create mock TTS audio
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
            
            # Verify output audio has content
            self.assertTrue(np.any(result != 0))
    
    def test_splice_audio_time_stretching(self):
        """Test audio time-stretching in splicing"""
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
            stretched_audio = np.random.rand(22050 * 2)  # Stretched to 2 seconds
            mock_stretch.return_value = stretched_audio
            
            output_audio = np.zeros(int(10 * 22050))
            result = splice_audio_with_segments(segments, output_audio, 22050)
            
            # Verify time-stretching was called with correct factor
            mock_stretch.assert_called_once()
            call_args = mock_stretch.call_args
            self.assertEqual(call_args[1]['rate'], 2.0)  # Should stretch by factor of 2
    
    def test_splice_audio_no_stretching_needed(self):
        """Test splicing when no time-stretching is needed"""
        segments = [
            {
                'type': 'tts',
                'start': 1.0,
                'end': 3.0,  # 2 seconds original duration
                'text': 'Hello world',
                'file': '/tmp/tts1.wav'
            }
        ]
        
        # Create TTS audio that's exactly 2 seconds (no stretching needed)
        tts_audio = np.random.rand(22050 * 2)  # 2 seconds
        
        with patch('librosa.load') as mock_load, \
             patch('librosa.effects.time_stretch') as mock_stretch:
            
            mock_load.return_value = (tts_audio, 22050)
            
            output_audio = np.zeros(int(10 * 22050))
            result = splice_audio_with_segments(segments, output_audio, 22050)
            
            # Verify time-stretching was NOT called
            mock_stretch.assert_not_called()
    
    def test_splice_audio_extreme_stretching(self):
        """Test splicing with extreme stretching factors (should be skipped)"""
        segments = [
            {
                'type': 'tts',
                'start': 1.0,
                'end': 1.1,  # 0.1 seconds original duration
                'text': 'Hello world',
                'file': '/tmp/tts1.wav'
            }
        ]
        
        # Create TTS audio that's 2 seconds (would need 20x stretching - too extreme)
        tts_audio = np.random.rand(22050 * 2)  # 2 seconds
        
        with patch('librosa.load') as mock_load, \
             patch('librosa.effects.time_stretch') as mock_stretch:
            
            mock_load.return_value = (tts_audio, 22050)
            
            output_audio = np.zeros(int(10 * 22050))
            result = splice_audio_with_segments(segments, output_audio, 22050)
            
            # Verify time-stretching was NOT called due to extreme factor
            mock_stretch.assert_not_called()
    
    def test_splice_audio_placement_accuracy(self):
        """Test that TTS audio is placed at correct timings"""
        segments = [
            {
                'type': 'tts',
                'start': 2.0,
                'end': 4.0,
                'text': 'Hello world',
                'file': '/tmp/tts1.wav'
            }
        ]
        
        # Create TTS audio that's exactly 2 seconds
        tts_audio = np.random.rand(22050 * 2)  # 2 seconds
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (tts_audio, 22050)
            
            output_audio = np.zeros(int(10 * 22050))
            result = splice_audio_with_segments(segments, output_audio, 22050)
            
            # Check that audio was placed at the correct position
            # Audio should start at 2 seconds (2 * 22050 samples)
            start_sample = int(2.0 * 22050)
            end_sample = start_sample + len(tts_audio)
            
            # The output should have content in the expected range
            self.assertTrue(np.any(result[start_sample:end_sample] != 0))
    
    def test_create_segments_from_srt(self):
        """Test creating segments from SRT data"""
        srt_data = [
            {'start': 1.0, 'end': 3.0, 'text': 'Hello world'},
            {'start': 4.0, 'end': 6.0, 'text': 'How are you today?'}
        ]
        
        segments = create_segments_from_srt(srt_data)
        
        self.assertEqual(len(segments), 2)
        
        # Check first segment
        self.assertEqual(segments[0]['type'], 'tts')
        self.assertEqual(segments[0]['start'], 1.0)
        self.assertEqual(segments[0]['end'], 3.0)
        self.assertEqual(segments[0]['text'], 'Hello world')
        
        # Check second segment
        self.assertEqual(segments[1]['type'], 'tts')
        self.assertEqual(segments[1]['start'], 4.0)
        self.assertEqual(segments[1]['end'], 6.0)
        self.assertEqual(segments[1]['text'], 'How are you today?')
    
    def test_parse_srt_edge_cases(self):
        """Test SRT parsing with edge cases"""
        # Test with empty content
        parsed = parse_srt("")
        self.assertEqual(parsed, [])
        
        # Test with single subtitle
        single_srt = """1
00:00:01,000 --> 00:00:03,000
Single subtitle"""
        
        parsed = parse_srt(single_srt)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]['text'], 'Single subtitle')
        
        # Test with subtitle containing newlines
        multiline_srt = """1
00:00:01,000 --> 00:00:03,000
Line 1
Line 2
Line 3"""
        
        parsed = parse_srt(multiline_srt)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]['text'], 'Line 1\nLine 2\nLine 3')
    
    def test_audio_quality_metrics(self):
        """Test audio quality metrics after processing"""
        # Create test segments
        segments = [
            {
                'type': 'tts',
                'start': 1.0,
                'end': 3.0,
                'text': 'Hello world',
                'file': '/tmp/tts1.wav'
            }
        ]
        
        # Create high-quality TTS audio
        tts_audio = np.random.rand(22050 * 2)
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (tts_audio, 22050)
            
            output_audio = np.zeros(int(10 * 22050))
            result = splice_audio_with_segments(segments, output_audio, 22050)
            
            # Check audio quality metrics
            # RMS (Root Mean Square) should be reasonable
            rms = np.sqrt(np.mean(result**2))
            self.assertGreater(rms, 0.0)
            self.assertLess(rms, 1.0)
            
            # Dynamic range should be reasonable
            dynamic_range = np.max(result) - np.min(result)
            self.assertGreater(dynamic_range, 0.0)
            self.assertLess(dynamic_range, 2.0)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of audio processing"""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create large segments
        segments = []
        for i in range(50):
            segments.append({
                'type': 'tts',
                'start': i * 2.0,
                'end': i * 2.0 + 1.5,
                'text': f'Segment {i+1}',
                'file': f'/tmp/tts{i}.wav'
            })
        
        # Create large output buffer
        output_audio = np.zeros(int(120 * 22050))  # 2 minutes
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (np.random.rand(22050 * 1), 22050)
            
            result = splice_audio_with_segments(segments, output_audio, 22050)
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 200MB)
            self.assertLess(memory_increase, 200 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main() 