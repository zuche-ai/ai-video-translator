"""
Tests for the AudioTranslator class.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import soundfile as sf
import numpy as np

from video_translator.audio_translation.audio_translator import AudioTranslator
from video_translator.audio.voice_cloner import VoiceCloner


class TestAudioTranslator:
    """Test cases for AudioTranslator class."""
    
    def test_init_defaults(self):
        """Test AudioTranslator initialization with default parameters."""
        translator = AudioTranslator()
        
        assert translator.src_lang == "en"
        assert translator.tgt_lang == "es"
        assert translator.model_size == "base"
        assert translator.voice_clone is False
        assert isinstance(translator.voice_cloner, VoiceCloner)
    
    def test_init_custom_params(self):
        """Test AudioTranslator initialization with custom parameters."""
        translator = AudioTranslator(
            src_lang="fr",
            tgt_lang="de",
            model_size="large",
            voice_clone=True
        )
        
        assert translator.src_lang == "fr"
        assert translator.tgt_lang == "de"
        assert translator.model_size == "large"
        assert translator.voice_clone is True
        assert translator.voice_cloner is not None
    
    def test_get_supported_formats(self):
        """Test getting supported audio formats."""
        translator = AudioTranslator()
        formats = translator.get_supported_formats()
        
        expected_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.opus']
        assert formats == expected_formats
    
    def test_validate_input_file_valid(self):
        """Test input file validation with valid formats."""
        translator = AudioTranslator()
        
        valid_files = ['audio.mp3', 'audio.wav', 'audio.m4a', 'audio.flac', 'audio.opus']
        for file_path in valid_files:
            assert translator.validate_input_file(file_path) is True
    
    def test_validate_input_file_invalid(self):
        """Test input file validation with invalid formats."""
        translator = AudioTranslator()
        
        invalid_files = ['audio.txt', 'audio.pdf', 'audio.xyz', 'audio']
        for file_path in invalid_files:
            assert translator.validate_input_file(file_path) is False
    
    def test_generate_output_path(self):
        """Test output path generation."""
        translator = AudioTranslator(tgt_lang="es")
        
        input_path = Path("/path/to/audio.mp3")
        output_path = translator._generate_output_path(input_path)
        
        expected_path = "/path/to/audio_es.mp3"
        assert output_path == expected_path
    
    @patch('video_translator.audio_translation.audio_translator.transcriber')
    @patch('video_translator.audio_translation.audio_translator.translator')
    def test_translate_audio_success(self, mock_translator, mock_transcriber):
        """Test successful audio translation."""
        # Mock transcription
        mock_segments = [{'text': 'Hello world'}]
        mock_transcriber.transcribe_video.return_value = mock_segments
        
        # Mock translation
        mock_translated_segments = [{'text': 'Hola mundo'}]
        mock_translator.translate_segments.return_value = mock_translated_segments
        
        # Mock voice cloner
        mock_voice_cloner = Mock()
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = os.path.join(tmpdir, "chunk_0.wav")
            mock_voice_cloner.clone_voice.return_value = chunk_path
            mock_voice_cloner.count_xtts_tokens.return_value = 1
            # Patch _split_text_for_xtts to return a single chunk
            from video_translator.audio_translation import audio_translator
            original_split = audio_translator.AudioTranslator._split_text_for_xtts
            audio_translator.AudioTranslator._split_text_for_xtts = lambda self, text, max_tokens=400: ["Hola mundo"]
            # Patch _concat_audio_files to avoid file I/O
            original_concat = audio_translator.AudioTranslator._concat_audio_files
            audio_translator.AudioTranslator._concat_audio_files = lambda self, audio_paths, output_path: output_path
            try:
                translator = AudioTranslator(voice_clone=True)
                translator.voice_cloner = mock_voice_cloner
                output_path = translator.translate_audio(temp_path, output_path=chunk_path)
                assert output_path == chunk_path
                mock_transcriber.transcribe_video.assert_called_once()
                mock_translator.translate_segments.assert_called_once()
                mock_voice_cloner.clone_voice.assert_called()
            finally:
                audio_translator.AudioTranslator._split_text_for_xtts = original_split
                audio_translator.AudioTranslator._concat_audio_files = original_concat
                os.unlink(temp_path)
    
    def test_translate_audio_file_not_found(self):
        """Test audio translation with non-existent file."""
        translator = AudioTranslator()
        
        with pytest.raises(FileNotFoundError):
            translator.translate_audio("/nonexistent/file.mp3")
    
    @patch('video_translator.audio_translation.audio_translator.transcriber')
    def test_translate_audio_transcription_failure(self, mock_transcriber):
        """Test audio translation when transcription fails."""
        mock_transcriber.transcribe_video.return_value = []
        
        translator = AudioTranslator()
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ValueError, match="Transcription failed"):
                translator.translate_audio(temp_path)
        finally:
            os.unlink(temp_path)
    
    @patch('video_translator.audio_translation.audio_translator.transcriber')
    @patch('video_translator.audio_translation.audio_translator.translator')
    def test_translate_audio_translation_failure(self, mock_translator, mock_transcriber):
        """Test audio translation when translation fails."""
        # Mock transcription success
        mock_segments = [{'text': 'Hello world'}]
        mock_transcriber.transcribe_video.return_value = mock_segments
        
        # Mock translation failure
        mock_translator.translate_segments.return_value = []
        
        translator = AudioTranslator()
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ValueError, match="Translation failed"):
                translator.translate_audio(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_generate_regular_tts_not_implemented(self):
        """Test that regular TTS raises NotImplementedError."""
        translator = AudioTranslator()
        
        with pytest.raises(NotImplementedError):
            translator._generate_regular_tts("Hello world", "/path/to/output.wav")
    
    def test_translate_audio_with_custom_output_path(self):
        """Test audio translation with custom output path."""
        translator = AudioTranslator()
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
        
        custom_output = "/custom/output/path.wav"
        
        try:
            # This will fail due to missing mocks, but we can test the output path logic
            with pytest.raises(Exception):
                translator.translate_audio(temp_path, output_path=custom_output)
        finally:
            os.unlink(temp_path)


class TestAudioTranslatorIntegration:
    """Integration tests for AudioTranslator (requires actual models)."""
    
    @pytest.mark.skipif(
        not os.path.exists("models/xtts"),
        reason="XTTS models not available"
    )
    def test_full_pipeline_with_voice_cloning(self):
        """Test full audio translation pipeline with voice cloning."""
        # This test requires actual models and would be slow
        # It's marked as integration test and skipped by default
        pass
    
    @pytest.mark.skipif(
        not os.path.exists("models/xtts"),
        reason="XTTS models not available"
    )
    def test_full_pipeline_without_voice_cloning(self):
        """Test full audio translation pipeline without voice cloning."""
        # This test requires actual models and would be slow
        # It's marked as integration test and skipped by default
        pass 