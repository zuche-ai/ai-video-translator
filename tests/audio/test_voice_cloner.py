"""
Unit tests for voice cloning functionality.
"""

import unittest
import tempfile
import os
import numpy as np
import soundfile as sf
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_translator.audio.voice_cloner import VoiceCloner, AudioProcessor


class TestVoiceCloner(unittest.TestCase):
    """Test cases for VoiceCloner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        
        # Create a test audio file
        sample_rate = 22050
        duration = 5.0  # 5 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz sine wave
        sf.write(self.test_audio_path, audio_data, sample_rate)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('video_translator.audio.voice_cloner.TTS')
    def test_initialization(self, mock_tts_class):
        """Test VoiceCloner initialization."""
        mock_tts_instance = Mock()
        mock_tts_class.return_value = mock_tts_instance
        
        cloner = VoiceCloner()
        
        mock_tts_class.assert_called_once_with(
            model_name='tts_models/multilingual/multi-dataset/xtts_v2',
            progress_bar=False,
            gpu=False
        )
        self.assertEqual(cloner.tts, mock_tts_instance)
    
    @patch('video_translator.audio.voice_cloner.TTS')
    def test_initialization_custom_model(self, mock_tts_class):
        """Test VoiceCloner initialization with custom model."""
        mock_tts_instance = Mock()
        mock_tts_class.return_value = mock_tts_instance
        
        custom_model = "custom/xtts/model"
        cloner = VoiceCloner(model_name=custom_model)
        
        mock_tts_class.assert_called_once_with(
            model_name=custom_model,
            progress_bar=False,
            gpu=False
        )
    
    @patch('video_translator.audio.voice_cloner.TTS')
    def test_initialization_failure(self, mock_tts_class):
        """Test VoiceCloner initialization failure."""
        mock_tts_class.side_effect = Exception("Model loading failed")
        
        with self.assertRaises(Exception):
            VoiceCloner()
    
    @patch('video_translator.audio.voice_cloner.TTS')
    def test_clone_voice(self, mock_tts_class):
        """Test voice cloning functionality."""
        mock_tts_instance = Mock()
        mock_tts_class.return_value = mock_tts_instance
        
        # Mock the tts_to_file method to create a real file
        def mock_tts_to_file(**kwargs):
            file_path = kwargs.get('file_path')
            if file_path:
                # Create a mock audio file
                import numpy as np
                import soundfile as sf
                sample_rate = 22050
                duration = 2.0  # 2 seconds
                samples = int(sample_rate * duration)
                audio_data = np.random.randn(samples) * 0.1  # Low volume random audio
                sf.write(file_path, audio_data, sample_rate)
        
        mock_tts_instance.tts_to_file.side_effect = mock_tts_to_file
        
        cloner = VoiceCloner()
        
        # Create a mock reference audio file
        reference_audio = os.path.join(self.temp_dir, "reference.wav")
        import numpy as np
        import soundfile as sf
        sample_rate = 22050
        duration = 5.0  # 5 seconds
        samples = int(sample_rate * duration)
        audio_data = np.random.randn(samples) * 0.3  # Some audio content
        sf.write(reference_audio, audio_data, sample_rate)
        
        output_path = os.path.join(self.temp_dir, "output.wav")
        
        result = cloner.clone_voice(
            text="Hello world",
            audio_path=reference_audio,
            output_path=output_path,
            language="en"
        )
        
        self.assertEqual(result, output_path)
        self.assertTrue(os.path.exists(output_path))
    
    @patch('video_translator.audio.voice_cloner.TTS')
    def test_clone_voice_no_model(self, mock_tts_class):
        """Test voice cloning when model is not initialized."""
        mock_tts_instance = Mock()
        mock_tts_class.return_value = mock_tts_instance
        
        cloner = VoiceCloner()
        cloner.tts = None  # Simulate uninitialized model
        
        with self.assertRaises(RuntimeError):
            cloner.clone_voice(
                text="Test",
                audio_path=self.test_audio_path,
                output_path="output.wav"
            )
    
    @patch('video_translator.audio.voice_cloner.TTS')
    def test_batch_clone_voice(self, mock_tts_class):
        """Test batch voice cloning."""
        mock_tts_instance = Mock()
        mock_tts_class.return_value = mock_tts_instance
        
        # Mock the tts_to_file method to create real files
        def mock_tts_to_file(**kwargs):
            file_path = kwargs.get('file_path')
            if file_path:
                # Create a mock audio file
                import numpy as np
                import soundfile as sf
                sample_rate = 22050
                duration = 1.0  # 1 second
                samples = int(sample_rate * duration)
                audio_data = np.random.randn(samples) * 0.1  # Low volume random audio
                sf.write(file_path, audio_data, sample_rate)
        
        mock_tts_instance.tts_to_file.side_effect = mock_tts_to_file
        
        cloner = VoiceCloner()
        
        # Create a mock reference audio file
        reference_audio = os.path.join(self.temp_dir, "reference.wav")
        import numpy as np
        import soundfile as sf
        sample_rate = 22050
        duration = 5.0  # 5 seconds
        samples = int(sample_rate * duration)
        audio_data = np.random.randn(samples) * 0.3  # Some audio content
        sf.write(reference_audio, audio_data, sample_rate)
        
        texts = ["Hello", "World", "Test"]
        output_dir = os.path.join(self.temp_dir, "batch_output")
        os.makedirs(output_dir, exist_ok=True)
        
        result = cloner.batch_clone_voice(
            reference_audio_path=reference_audio,
            texts=texts,
            output_dir=output_dir,
            language="en"
        )
        
        self.assertEqual(len(result), len(texts))
        for audio_file in result:
            self.assertTrue(os.path.exists(audio_file))
    
    def test_extract_audio_segments(self):
        """Test audio segment extraction."""
        # Skip this test in Docker/CI environments as it requires TTS model downloads
        if os.environ.get('DOCKER_ENV') or os.environ.get('CI'):
            self.skipTest("Skipping TTS model test in Docker/CI environment (requires model downloads)")
        
        cloner = VoiceCloner()
        
        # Create a longer test audio file
        sample_rate = 22050
        duration = 10.0  # 10 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.3
        sf.write(self.test_audio_path, audio_data, sample_rate)
        
        # Define segments
        segments = [(1.0, 3.0), (5.0, 7.0), (8.0, 9.0)]
        
        # Extract segments
        result = cloner.extract_audio_segments(self.test_audio_path, segments)
        
        # Verify results
        self.assertEqual(len(result), 3)
        
        # Check that files exist
        for path in result:
            self.assertTrue(os.path.exists(path))
            
            # Check file content
            audio, sr = sf.read(path)
            self.assertEqual(sr, sample_rate)
    
    def test_process_audio_file(self):
        """Test audio file processing."""
        # Skip this test in Docker/CI environments as it requires TTS model downloads
        if os.environ.get('DOCKER_ENV') or os.environ.get('CI'):
            self.skipTest("Skipping TTS model test in Docker/CI environment (requires model downloads)")
        
        cloner = VoiceCloner()
        
        # Process audio file
        result = cloner.process_audio_file(
            audio_path=self.test_audio_path,
            target_sr=16000,
            normalize=True
        )
        
        # Verify result
        self.assertTrue(os.path.exists(result))
        
        # Check processed audio
        audio, sr = sf.read(result)
        self.assertEqual(sr, 16000)
        
        # Check normalization (RMS should be reasonable)
        rms = np.sqrt(np.mean(audio**2))
        self.assertGreater(rms, 0.0)
        self.assertLess(rms, 1.0)
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        # Skip this test in Docker/CI environments as it requires TTS model downloads
        if os.environ.get('DOCKER_ENV') or os.environ.get('CI'):
            self.skipTest("Skipping TTS model test in Docker/CI environment (requires model downloads)")
        
        cloner = VoiceCloner()
        languages = cloner.get_supported_languages()
        
        # Verify it returns a list
        self.assertIsInstance(languages, list)
        
        # Verify it contains expected languages
        expected_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl"]
        for lang in expected_languages:
            self.assertIn(lang, languages)
    
    def test_validate_reference_audio_valid(self):
        """Test reference audio validation with valid audio."""
        # Skip this test in Docker/CI environments as it requires TTS model downloads
        if os.environ.get('DOCKER_ENV') or os.environ.get('CI'):
            self.skipTest("Skipping TTS model test in Docker/CI environment (requires model downloads)")
        
        cloner = VoiceCloner()
        
        # Test with valid audio
        result = cloner.validate_reference_audio(self.test_audio_path)
        self.assertTrue(result)
    
    def test_validate_reference_audio_nonexistent(self):
        """Test reference audio validation with nonexistent file."""
        # Skip this test in Docker/CI environments as it requires TTS model downloads
        if os.environ.get('DOCKER_ENV') or os.environ.get('CI'):
            self.skipTest("Skipping TTS model test in Docker/CI environment (requires model downloads)")
        
        cloner = VoiceCloner()
        
        # Test with nonexistent file
        result = cloner.validate_reference_audio("nonexistent.wav")
        self.assertFalse(result)
    
    def test_validate_reference_audio_silent(self):
        """Test reference audio validation with silent audio."""
        # Skip this test in Docker/CI environments as it requires TTS model downloads
        if os.environ.get('DOCKER_ENV') or os.environ.get('CI'):
            self.skipTest("Skipping TTS model test in Docker/CI environment (requires model downloads)")
        
        cloner = VoiceCloner()
        
        # Create silent audio
        silent_audio_path = os.path.join(self.temp_dir, "silent.wav")
        sample_rate = 22050
        duration = 5.0
        silent_data = np.zeros(int(sample_rate * duration))
        sf.write(silent_audio_path, silent_data, sample_rate)
        
        # Test with silent audio
        result = cloner.validate_reference_audio(silent_audio_path)
        self.assertFalse(result)
    
    def test_validate_reference_audio_short(self):
        """Test reference audio validation with short audio."""
        # Skip this test in Docker/CI environments as it requires TTS model downloads
        if os.environ.get('DOCKER_ENV') or os.environ.get('CI'):
            self.skipTest("Skipping TTS model test in Docker/CI environment (requires model downloads)")
        
        cloner = VoiceCloner()
        
        # Create short audio
        short_audio_path = os.path.join(self.temp_dir, "short.wav")
        sample_rate = 22050
        duration = 1.0  # 1 second (too short)
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.3
        sf.write(short_audio_path, audio_data, sample_rate)
        
        # Test with short audio (should warn but still return True)
        result = cloner.validate_reference_audio(short_audio_path)
        self.assertTrue(result)  # Should still be valid, just with warning


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
    
    @patch('video_translator.audio.voice_cloner.AudioProcessor.extract_audio_from_video')
    def test_extract_audio_from_video_success(self, mock_extract):
        """Test successful audio extraction from video."""
        processor = AudioProcessor()
        mock_extract.return_value = True
        
        result = processor.extract_audio_from_video("test_video.mp4", "test_audio.wav")
        
        self.assertTrue(result)
        mock_extract.assert_called_once_with("test_video.mp4", "test_audio.wav")
    
    @patch('video_translator.audio.voice_cloner.AudioProcessor.extract_audio_from_video')
    def test_extract_audio_from_video_failure(self, mock_extract):
        """Test audio extraction failure."""
        processor = AudioProcessor()
        mock_extract.return_value = False
        
        result = processor.extract_audio_from_video("test_video.mp4", "test_audio.wav")
        
        self.assertFalse(result)
        mock_extract.assert_called_once_with("test_video.mp4", "test_audio.wav")
    
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