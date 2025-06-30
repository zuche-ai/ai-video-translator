"""
Voice cloning module using Coqui TTS for local voice cloning and audio generation.
"""

import os
import tempfile
import logging
from typing import Optional, Tuple, List
import numpy as np
import soundfile as sf
import librosa
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

logger = logging.getLogger(__name__)


class VoiceCloner:
    """
    Voice cloning class using Coqui TTS for local voice cloning.
    """
    
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        """
        Initialize the voice cloner with Coqui TTS.
        
        Args:
            model_name: Name of the TTS model to use
        """
        self.model_name = model_name
        self.tts = None
        self.voice_profile = None
        self.sample_rate = 22050
        
        try:
            # Initialize TTS model
            self.tts = TTS(model_name)
            logger.info(f"Initialized TTS model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            raise
    
    def extract_voice_profile(self, audio_file: str, duration: float = 10.0) -> bool:
        """
        Extract voice profile from an audio file.
        
        Args:
            audio_file: Path to the audio file
            duration: Duration of audio to use for voice profiling (seconds)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load audio and extract a sample for voice profiling
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Use the first N seconds for voice profiling
            sample_length = int(duration * sr)
            if len(audio) > sample_length:
                audio_sample = audio[:sample_length]
            else:
                audio_sample = audio
            
            # Save the voice profile sample
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_sample, sr)
                self.voice_profile = temp_file.name
            
            logger.info(f"Voice profile extracted from {audio_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract voice profile: {e}")
            return False
    
    def clone_voice(self, text: str, output_file: str, language: str = "en") -> bool:
        """
        Generate speech with cloned voice.
        
        Args:
            text: Text to convert to speech
            output_file: Output audio file path
            language: Language code for the text
            
        Returns:
            True if successful, False otherwise
        """
        if not self.voice_profile:
            logger.error("No voice profile available. Call extract_voice_profile first.")
            return False
        
        try:
            # Generate speech with cloned voice
            self.tts.tts_to_file(
                text=text,
                speaker_wav=self.voice_profile,
                language=language,
                file_path=output_file
            )
            
            logger.info(f"Generated cloned voice audio: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate cloned voice: {e}")
            return False
    
    def batch_clone_voice(self, texts: List[str], output_dir: str, 
                         language: str = "en", prefix: str = "segment_") -> List[str]:
        """
        Generate multiple audio segments with cloned voice.
        
        Args:
            texts: List of texts to convert to speech
            output_dir: Directory to save output files
            language: Language code for the texts
            prefix: Prefix for output filenames
            
        Returns:
            List of generated audio file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        for i, text in enumerate(texts):
            output_file = os.path.join(output_dir, f"{prefix}{i:04d}.wav")
            if self.clone_voice(text, output_file, language):
                output_files.append(output_file)
            else:
                logger.warning(f"Failed to generate audio for text {i}")
        
        return output_files
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.voice_profile and os.path.exists(self.voice_profile):
            try:
                os.unlink(self.voice_profile)
                logger.info("Cleaned up voice profile")
            except Exception as e:
                logger.warning(f"Failed to cleanup voice profile: {e}")


class AudioProcessor:
    """
    Audio processing utilities for voice cloning workflow.
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def extract_audio_from_video(self, video_file: str, output_file: str) -> bool:
        """
        Extract audio from video file.
        
        Args:
            video_file: Path to input video file
            output_file: Path to output audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import ffmpeg
            
            # Extract audio using ffmpeg
            stream = ffmpeg.input(video_file)
            stream = ffmpeg.output(stream, output_file, acodec='pcm_s16le', 
                                 ar=self.sample_rate, ac=1)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            logger.info(f"Extracted audio from {video_file} to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return False
    
    def merge_audio_segments(self, audio_files: List[str], output_file: str, 
                           timestamps: List[float]) -> bool:
        """
        Merge audio segments with proper timing.
        
        Args:
            audio_files: List of audio file paths
            timestamps: List of start timestamps for each segment
            output_file: Output merged audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load all audio segments
            segments = []
            for audio_file in audio_files:
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                segments.append(audio)
            
            # Calculate total duration needed
            max_duration = max(timestamps) + max(len(seg) for seg in segments) / self.sample_rate
            total_samples = int(max_duration * self.sample_rate)
            
            # Create output array
            output_audio = np.zeros(total_samples)
            
            # Place segments at their timestamps
            for i, (segment, timestamp) in enumerate(zip(segments, timestamps)):
                start_sample = int(timestamp * self.sample_rate)
                end_sample = start_sample + len(segment)
                
                if end_sample <= len(output_audio):
                    output_audio[start_sample:end_sample] = segment
                else:
                    # Truncate if segment goes beyond total duration
                    output_audio[start_sample:] = segment[:len(output_audio) - start_sample]
            
            # Save merged audio
            sf.write(output_file, output_audio, self.sample_rate)
            
            logger.info(f"Merged {len(audio_files)} audio segments to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge audio segments: {e}")
            return False
    
    def overlay_audio(self, original_audio: str, translated_audio: str, 
                     output_file: str, original_volume: float = 0.3) -> bool:
        """
        Overlay translated audio on original audio.
        
        Args:
            original_audio: Path to original audio file
            translated_audio: Path to translated audio file
            output_file: Path to output file
            original_volume: Volume of original audio (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load audio files
            orig_audio, sr1 = librosa.load(original_audio, sr=self.sample_rate)
            trans_audio, sr2 = librosa.load(translated_audio, sr=self.sample_rate)
            
            # Ensure same length
            max_length = max(len(orig_audio), len(trans_audio))
            orig_audio = np.pad(orig_audio, (0, max_length - len(orig_audio)))
            trans_audio = np.pad(trans_audio, (0, max_length - len(trans_audio)))
            
            # Mix audio with specified volumes
            mixed_audio = (orig_audio * original_volume) + trans_audio
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val
            
            # Save mixed audio
            sf.write(output_file, mixed_audio, self.sample_rate)
            
            logger.info(f"Created audio overlay: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to overlay audio: {e}")
            return False 