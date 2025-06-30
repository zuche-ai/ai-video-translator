"""
Unit tests for voice cloning functionality.
"""

import unittest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_translator.voice_cloner import VoiceCloner, AudioProcessor


class TestVoiceCloner(unittest.TestCase):
    """Test cases for VoiceCloner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_file = os.path.join(self.temp_dir, "test_audio.wav")
        
        # Create a simple test audio file
        sample_rate = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.1  # Simple sine wave
        
        import soundfile as sf
        sf.write(self.test_audio_file, audio_data, sample_rate)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('video_translator.voice_cloner.TTS')
    def test_voice_cloner_initialization(self, mock_tts):
        """Test VoiceCloner initialization."""
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance
        
        cloner = VoiceCloner()
        
        self.assertIsNotNone(cloner.tts)
        self.assertIsNone(cloner.voice_profile)
        self.assertEqual(cloner.sample_rate, 22050)
        mock_tts.assert_called_once()
    
    @patch('video_translator.voice_cloner.TTS')
    def test_voice_cloner_initialization_failure(self, mock_tts):
        """Test VoiceCloner initialization failure."""
        mock_tts.side_effect = Exception("TTS initialization failed")
        
        with self.assertRaises(Exception):
            VoiceCloner()
    
    @patch('video_translator.voice_cloner.TTS')
    @patch('video_translator.voice_cloner.librosa.load')
    @patch('video_translator.voice_cloner.sf.write')
    def test_extract_voice_profile_success(self, mock_sf_write, mock_librosa_load, mock_tts):
        """Test successful voice profile extraction."""
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance
        
        # Mock librosa.load to return test audio data
        sample_rate = 22050
        duration = 5.0
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        mock_librosa_load.return_value = (audio_data, sample_rate)
        
        cloner = VoiceCloner()
        result = cloner.extract_voice_profile(self.test_audio_file, duration=3.0)
        
        self.assertTrue(result)
        self.assertIsNotNone(cloner.voice_profile)
        mock_librosa_load.assert_called_once()
        mock_sf_write.assert_called_once()
    
    @patch('video_translator.voice_cloner.TTS')
    @patch('video_translator.voice_cloner.librosa.load')
    def test_extract_voice_profile_failure(self, mock_librosa_load, mock_tts):
        """Test voice profile extraction failure."""
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance
        mock_librosa_load.side_effect = Exception("Audio loading failed")
        
        cloner = VoiceCloner()
        result = cloner.extract_voice_profile(self.test_audio_file)
        
        self.assertFalse(result)
    
    @patch('video_translator.voice_cloner.TTS')
    def test_clone_voice_no_profile(self, mock_tts):
        """Test voice cloning without voice profile."""
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance
        
        cloner = VoiceCloner()
        output_file = os.path.join(self.temp_dir, "output.wav")
        
        result = cloner.clone_voice("Hello world", output_file)
        
        self.assertFalse(result)
    
    @patch('video_translator.voice_cloner.TTS')
    @patch('video_translator.voice_cloner.librosa.load')
    @patch('video_translator.voice_cloner.sf.write')
    def test_clone_voice_success(self, mock_sf_write, mock_librosa_load, mock_tts):
        """Test successful voice cloning."""
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance
        
        # Mock librosa.load
        sample_rate = 22050
        duration = 5.0
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        mock_librosa_load.return_value = (audio_data, sample_rate)
        
        cloner = VoiceCloner()
        
        # First extract voice profile
        cloner.extract_voice_profile(self.test_audio_file)
        
        # Then test voice cloning
        output_file = os.path.join(self.temp_dir, "output.wav")
        result = cloner.clone_voice("Hello world", output_file, "en")
        
        # Since we're mocking TTS, this should work
        self.assertTrue(result)
        mock_tts_instance.tts_to_file.assert_called_once()
    
    @patch('video_translator.voice_cloner.TTS')
    @patch('video_translator.voice_cloner.librosa.load')
    @patch('video_translator.voice_cloner.sf.write')
    def test_batch_clone_voice(self, mock_sf_write, mock_librosa_load, mock_tts):
        """Test batch voice cloning."""
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance
        
        # Mock librosa.load
        sample_rate = 22050
        duration = 5.0
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        mock_librosa_load.return_value = (audio_data, sample_rate)
        
        cloner = VoiceCloner()
        cloner.extract_voice_profile(self.test_audio_file)
        
        texts = ["Hello", "World", "Test"]
        output_dir = os.path.join(self.temp_dir, "batch_output")
        
        result = cloner.batch_clone_voice(texts, output_dir, "en")
        
        self.assertEqual(len(result), 3)
        self.assertEqual(mock_tts_instance.tts_to_file.call_count, 3)
    
    @patch('video_translator.voice_cloner.TTS')
    def test_cleanup(self, mock_tts):
        """Test cleanup functionality."""
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance
        
        cloner = VoiceCloner()
        
        # Create a temporary voice profile file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            cloner.voice_profile = temp_file.name
        
        cloner.cleanup()
        
        # File should be deleted
        self.assertFalse(os.path.exists(cloner.voice_profile))


class TestAudioProcessor(unittest.TestCase):
    """Test cases for AudioProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = AudioProcessor()
        
        # Create test audio files
        self.test_audio1 = os.path.join(self.temp_dir, "test1.wav")
        self.test_audio2 = os.path.join(self.temp_dir, "test2.wav")
        
        # Create simple test audio
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data1 = np.sin(2 * np.pi * 440 * t) * 0.1
        audio_data2 = np.sin(2 * np.pi * 880 * t) * 0.1
        
        import soundfile as sf
        sf.write(self.test_audio1, audio_data1, sample_rate)
        sf.write(self.test_audio2, audio_data2, sample_rate)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('video_translator.voice_cloner.ffmpeg')
    def test_extract_audio_from_video_success(self, mock_ffmpeg):
        """Test successful audio extraction from video."""
        mock_input = Mock()
        mock_output = Mock()
        mock_run = Mock()
        
        mock_ffmpeg.input.return_value = mock_input
        mock_input.audio = Mock()
        mock_ffmpeg.output.return_value = mock_output
        mock_ffmpeg.run = mock_run
        
        video_file = "test_video.mp4"
        output_file = os.path.join(self.temp_dir, "extracted.wav")
        
        result = self.processor.extract_audio_from_video(video_file, output_file)
        
        self.assertTrue(result)
        mock_ffmpeg.input.assert_called_once_with(video_file)
        mock_ffmpeg.run.assert_called_once()
    
    @patch('video_translator.voice_cloner.ffmpeg')
    def test_extract_audio_from_video_failure(self, mock_ffmpeg):
        """Test audio extraction failure."""
        mock_ffmpeg.input.side_effect = Exception("FFmpeg failed")
        
        video_file = "test_video.mp4"
        output_file = os.path.join(self.temp_dir, "extracted.wav")
        
        result = self.processor.extract_audio_from_video(video_file, output_file)
        
        self.assertFalse(result)
    
    def test_merge_audio_segments_success(self):
        """Test successful audio segment merging."""
        audio_files = [self.test_audio1, self.test_audio2]
        timestamps = [0.0, 2.5]  # Start times for each segment
        output_file = os.path.join(self.temp_dir, "merged.wav")
        
        result = self.processor.merge_audio_segments(audio_files, output_file, timestamps)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_file))
    
    def test_merge_audio_segments_failure(self):
        """Test audio segment merging failure."""
        audio_files = ["nonexistent1.wav", "nonexistent2.wav"]
        timestamps = [0.0, 1.0]
        output_file = os.path.join(self.temp_dir, "merged.wav")
        
        result = self.processor.merge_audio_segments(audio_files, output_file, timestamps)
        
        self.assertFalse(result)
    
    def test_overlay_audio_success(self):
        """Test successful audio overlay."""
        output_file = os.path.join(self.temp_dir, "overlay.wav")
        
        result = self.processor.overlay_audio(
            self.test_audio1, self.test_audio2, output_file, original_volume=0.5
        )
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_file))
    
    def test_overlay_audio_failure(self):
        """Test audio overlay failure."""
        output_file = os.path.join(self.temp_dir, "overlay.wav")
        
        result = self.processor.overlay_audio(
            "nonexistent1.wav", "nonexistent2.wav", output_file
        )
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main() 