"""
Audio Translator

Handles the complete pipeline for audio translation:
1. Transcribe audio to text
2. Translate the text
3. Generate new audio (regular TTS or voice cloning)
4. Save as new audio file
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import shutil
import soundfile as sf
import re
import numpy as np

from ..core import transcriber
from ..core import translator
from ..audio.voice_cloner import VoiceCloner


class AudioTranslator:
    """Main class for audio translation pipeline."""
    
    def __init__(self, 
                 src_lang: str = "en",
                 tgt_lang: str = "es",
                 model_size: str = "base",
                 voice_clone: bool = False):
        """
        Initialize the AudioTranslator.
        
        Args:
            src_lang: Source language code (e.g., "en", "es", "fr")
            tgt_lang: Target language code
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            voice_clone: Whether to use voice cloning instead of regular TTS
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_size = model_size
        self.voice_clone = voice_clone
        
        # Initialize components
        self.voice_cloner = VoiceCloner()
        self.model_size = model_size
        
        self.logger = logging.getLogger(__name__)
    
    def _split_text_for_xtts(self, text: str, max_tokens: int = 400) -> list:
        """
        Split text into chunks <= max_tokens for XTTS, using the actual tokenizer.
        Defensive: re-initialize VoiceCloner if tts or tts_model is missing.
        """
        if not hasattr(self, 'voice_cloner') or self.voice_cloner is None:
            self.logger.warning("VoiceCloner not initialized, initializing now.")
            self.voice_cloner = VoiceCloner()
        if not hasattr(self.voice_cloner, 'tts') or self.voice_cloner.tts is None:
            self.logger.warning("VoiceCloner.tts not initialized, re-initializing model.")
            self.voice_cloner._initialize_model()
        tts = self.voice_cloner.tts
        if not hasattr(tts, 'synthesizer') or tts.synthesizer is None or not hasattr(tts.synthesizer, 'tts_model') or tts.synthesizer.tts_model is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[DEBUG] _split_text_for_xtts: self.voice_cloner.tts type: {type(tts)}")
            logger.error(f"[DEBUG] _split_text_for_xtts: self.voice_cloner.tts: {tts}")
            if hasattr(tts, 'synthesizer'):
                logger.error(f"[DEBUG] _split_text_for_xtts: self.voice_cloner.tts.synthesizer: {tts.synthesizer}")
                if hasattr(tts.synthesizer, 'tts_model'):
                    logger.error(f"[DEBUG] _split_text_for_xtts: self.voice_cloner.tts.synthesizer.tts_model: {tts.synthesizer.tts_model}")
                else:
                    logger.error(f"[DEBUG] _split_text_for_xtts: self.voice_cloner.tts.synthesizer has no tts_model attribute")
            else:
                logger.error(f"[DEBUG] _split_text_for_xtts: self.voice_cloner.tts has no synthesizer attribute")
            self.logger.error("VoiceCloner.tts.synthesizer.tts_model is missing after initialization!")
            raise RuntimeError("XTTS model not initialized or missing tts_model.")
        tokenizer = tts.synthesizer.tts_model.tokenizer
        language = self.tgt_lang
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current = ""
        for sent in sentences:
            candidate = (current + " " + sent).strip() if current else sent
            if len(tokenizer.encode(candidate, language)) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If sentence itself is too long, split by words and tokens
                words = sent.split()
                temp = ""
                for word in words:
                    candidate = (temp + " " + word).strip() if temp else word
                    if len(tokenizer.encode(candidate, language)) <= max_tokens:
                        temp = candidate
                    else:
                        if temp:
                            chunks.append(temp)
                        temp = word
                if temp:
                    chunks.append(temp)
                current = ""
        if current:
            chunks.append(current)
        return chunks

    def _concat_audio_files(self, audio_paths, output_path):
        """
        Concatenate multiple audio files into one output file.
        """
        data = []
        samplerate = None
        for path in audio_paths:
            d, sr = sf.read(path)
            if samplerate is None:
                samplerate = sr
            data.append(d)
        data = np.concatenate(data)
        sf.write(output_path, data, samplerate)
        return output_path

    def translate_audio(self, 
                       input_path: str, 
                       output_path: Optional[str] = None,
                       voice_clone: Optional[bool] = None) -> str:
        """
        Translate an audio file to a new audio file in the target language.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output audio file (auto-generated if None)
            voice_clone: Override voice cloning setting for this translation
            
        Returns:
            Path to the translated audio file
        """
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
        # Determine voice cloning setting
        use_voice_clone = voice_clone if voice_clone is not None else self.voice_clone
        
        # Generate output path if not provided
        if output_path is None:
            output_path = self._generate_output_path(input_path_obj)
        
        self.logger.info(f"Starting audio translation: {input_path} -> {output_path}")
        self.logger.info(f"Source: {self.src_lang}, Target: {self.tgt_lang}, Voice clone: {use_voice_clone}")
        
        try:
            # Step 1: Transcribe audio
            self.logger.info("Step 1: Transcribing audio...")
            segments = transcriber.transcribe_video(str(input_path), language=self.src_lang)
            
            # Extract text from segments
            transcription = " ".join([segment.get('text', '') for segment in segments])
            
            if not transcription:
                raise ValueError("Transcription failed - no text was extracted")
            
            self.logger.info(f"Transcription completed: {len(transcription)} characters")
            
            # Step 2: Translate text
            self.logger.info("Step 2: Translating text...")
            # Create a single segment for translation
            segments_to_translate = [{'text': transcription}]
            translated_segments = translator.translate_segments(segments_to_translate, self.src_lang, self.tgt_lang)
            translated_text = translated_segments[0]['text'] if translated_segments else ""
            
            if not translated_text:
                raise ValueError("Translation failed - no text was translated")
            
            self.logger.info(f"Translation completed: {len(translated_text)} characters")
            
            # Step 3: Generate audio
            self.logger.info("Step 3: Generating audio...")
            if use_voice_clone:
                if self.voice_cloner is None:
                    self.voice_cloner = VoiceCloner()
                # Split text for XTTS
                chunks = self._split_text_for_xtts(translated_text, max_tokens=400)
                self.logger.info(f"Voice cloning {len(chunks)} chunks due to XTTS token limit...")
                temp_audio_files = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    for idx, chunk in enumerate(chunks):
                        chunk_audio_path = os.path.join(tmpdir, f"chunk_{idx}.wav")
                        self.voice_cloner.clone_voice(
                            text=chunk,
                            audio_path=str(input_path),
                            output_path=chunk_audio_path,
                            language=self.tgt_lang
                        )
                        temp_audio_files.append(chunk_audio_path)
                    # Concatenate all chunk audio files
                    self._concat_audio_files(temp_audio_files, str(output_path))
                audio_path = output_path
            else:
                audio_path = self._generate_regular_tts(translated_text, output_path)
            
            self.logger.info(f"Audio generation completed: {audio_path}")
            
            return str(audio_path)
            
        except Exception as e:
            self.logger.error(f"Audio translation failed: {str(e)}")
            raise
    
    def _generate_output_path(self, input_path: Path) -> str:
        """Generate output path based on input path and target language."""
        stem = input_path.stem
        suffix = input_path.suffix
        return str(input_path.parent / f"{stem}_{self.tgt_lang}{suffix}")
    
    def _generate_regular_tts(self, text: str, output_path: str) -> str:
        """
        Generate regular TTS audio (placeholder - implement with your preferred TTS library).
        
        Args:
            text: Text to convert to speech
            output_path: Path for output audio file
            
        Returns:
            Path to generated audio file
        """
        # TODO: Implement regular TTS using a library like gTTS, pyttsx3, or Coqui TTS
        # For now, this is a placeholder that raises an error
        raise NotImplementedError(
            "Regular TTS not yet implemented. Use voice_clone=True for now, "
            "or implement TTS functionality in _generate_regular_tts method."
        )
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio input formats."""
        return ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.opus']
    
    def validate_input_file(self, file_path: str) -> bool:
        """Validate that the input file is a supported audio format."""
        path = Path(file_path)
        return path.suffix.lower() in self.get_supported_formats() 