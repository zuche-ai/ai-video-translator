"""
Tests for the audio translator functionality.
"""

import os
import tempfile
import unittest
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import video_translator
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from video_translator.audio_translation.audio_translator import AudioTranslator


class TestAudioTranslator(unittest.TestCase):
    """Test cases for AudioTranslator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.translator = AudioTranslator(temp_dir=self.temp_dir)
        
        # Create a test audio file
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        self.create_test_audio_file(self.test_audio_path)
        
        # Create a test reference audio file
        self.reference_audio_path = os.path.join(self.temp_dir, "reference_audio.wav")
        self.create_test_audio_file(self.reference_audio_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_audio_file(self, file_path, duration=5.0, sample_rate=16000):
        """Create a test audio file with some speech-like content."""
        # Generate a simple sine wave with some variation to simulate speech
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Mix of frequencies to simulate speech
        audio = (0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
                0.2 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency
                0.1 * np.sin(2 * np.pi * 2000 * t))  # High frequency
        
        # Add some silence at the beginning and end
        silence_samples = int(0.5 * sample_rate)  # 0.5 seconds of silence
        audio = np.concatenate([
            np.zeros(silence_samples),
            audio,
            np.zeros(silence_samples)
        ])
        
        # Normalize and save
        audio = audio / np.max(np.abs(audio)) * 0.8
        sf.write(file_path, audio, sample_rate)
    
    @patch('video_translator.core.transcriber.transcribe_video')
    @patch('video_translator.core.translator.translate_segments')
    @patch('video_translator.audio.voice_cloner.VoiceCloner')
    def test_translate_audio_basic(self, mock_voice_cloner, mock_translate, mock_transcribe):
        """Test basic audio translation functionality."""
        # Mock the transcription result
        mock_segments = [
            {'start': 0.5, 'end': 2.0, 'text': 'Hello world'},
            {'start': 2.5, 'end': 4.0, 'text': 'How are you'}
        ]
        mock_transcribe.return_value = mock_segments
        
        # Mock the translation result
        mock_translated_segments = [
            {'start': 0.5, 'end': 2.0, 'text': 'Hola mundo'},
            {'start': 2.5, 'end': 4.0, 'text': 'Como estas'}
        ]
        mock_translate.return_value = mock_translated_segments
        
        # Mock the voice cloner
        mock_cloner_instance = MagicMock()
        mock_cloner_instance.batch_clone_voice.return_value = [
            os.path.join(self.temp_dir, "voice_0000.wav"),
            os.path.join(self.temp_dir, "voice_0001.wav")
        ]
        mock_voice_cloner.return_value = mock_cloner_instance
        
        # Create test output files
        voice_file1 = os.path.join(self.temp_dir, "voice_0000.wav")
        voice_file2 = os.path.join(self.temp_dir, "voice_0001.wav")
        self.create_test_audio_file(voice_file1, duration=1.0)
        self.create_test_audio_file(voice_file2, duration=1.0)
        
        # Test the translation
        output_path = os.path.join(self.temp_dir, "output.wav")
        
        with patch('subprocess.run') as mock_subprocess:
            # Mock ffmpeg calls
            mock_subprocess.return_value.returncode = 0
            
            result = self.translator.translate_audio(
                audio_path=self.test_audio_path,
                src_lang="en",
                tgt_lang="es",
                reference_audio_path=self.reference_audio_path,
                output_path=output_path
            )
            
            # Verify the result
            self.assertEqual(result, output_path)
            
            # Verify mocks were called
            mock_transcribe.assert_called_once()
            mock_translate.assert_called_once()
            mock_cloner_instance.batch_clone_voice.assert_called_once()
    
    def test_analyze_voice_activity(self):
        """Test voice activity detection."""
        # Create segments for testing
        segments = [
            {'start': 0.0, 'end': 1.0, 'text': 'Test segment 1'},
            {'start': 1.0, 'end': 2.0, 'text': 'Test segment 2'}
        ]
        
        # Test VAD analysis
        voice_segments, non_voice_segments = self.translator._analyze_voice_activity(
            self.test_audio_path, segments
        )
        
        # Should return lists (actual content depends on VAD analysis)
        self.assertIsInstance(voice_segments, list)
        self.assertIsInstance(non_voice_segments, list)
        
        # All segments should be classified as either voice or non-voice
        total_segments = len(voice_segments) + len(non_voice_segments)
        self.assertEqual(total_segments, len(segments))
    

    
    def test_cleanup_temp_files(self):
        """Test temporary file cleanup."""
        # Create a temporary directory
        temp_base = os.path.join(self.temp_dir, "test_cleanup")
        test_file = os.path.join(temp_base, "test.txt")
        
        os.makedirs(temp_base, exist_ok=True)
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Verify file exists
        self.assertTrue(os.path.exists(test_file))
        
        # Test cleanup
        self.translator._cleanup_temp_files(temp_base)
        
        # Verify directory is removed
        self.assertFalse(os.path.exists(temp_base))


if __name__ == '__main__':
    unittest.main() 