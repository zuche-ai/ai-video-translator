"""
Voice cloning module using alternative TTS approaches for local voice cloning and audio generation.
"""

import os
import tempfile
import logging
from typing import Optional, Tuple, List
import numpy as np
import soundfile as sf
import librosa
from TTS.api import TTS
import torch
import webrtcvad
import traceback
import subprocess

logger = logging.getLogger(__name__)


class VoiceCloner:
    """Voice cloning using Coqui XTTS for true voice cloning."""
    
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        """
        Initialize the voice cloner with XTTS model.
        
        Args:
            model_name: XTTS model name to use
        """
        self.model_name = model_name
        self.tts = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the XTTS model with proper error handling."""
        try:
            # Set environment variables
            os.environ["COQUI_TOS_AGREED"] = "1"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
            
            # Clear GPU memory and garbage collect
            import gc
            gc.collect()
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fix PyTorch 2.6 compatibility issue with weights_only
            original_torch_load = torch.load
            def safe_torch_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            torch.load = safe_torch_load
            
            # Load model directly (will download if not present)
            self.tts = TTS(
                model_name=self.model_name,
                progress_bar=False,
                gpu=False  # Explicitly disable GPU
            )
            
            logger.info(f"XTTS model '{self.model_name}' loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Voice cloning initialization failed: {e}")
    
    def _get_supported_languages(self, model_name):
        try:
            # Accept Coqui TTS license terms
            os.environ["COQUI_TOS_AGREED"] = "1"
            tts = TTS(model_name=model_name, progress_bar=False)
            langs = set()
            if hasattr(tts, 'languages') and tts.languages:
                langs = set(list(tts.languages))
            elif hasattr(tts, 'list_languages') and callable(tts.list_languages):
                langs = set(list(tts.list_languages()))
            return langs
        except Exception as e:
            logger.error(f"Could not get supported languages for {model_name}: {e}")
            return set()

    def _get_fallback_tts(self, language):
        """Get a fallback TTS model that supports the requested language."""
        # Accept Coqui TTS license terms
        os.environ["COQUI_TOS_AGREED"] = "1"
        fallback_models = [
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/multilingual/multi-dataset/xtts_v1.1",
            "tts_models/multilingual/multi-dataset/your_tts",
            "tts_models/multilingual/multi-dataset/coqui_studio_TTS",
            "tts_models/en/ljspeech/tacotron2-DDC"  # last resort
        ]
        for model_name in fallback_models:
            langs = self._get_supported_languages(model_name)
            if not langs or language in langs or language.lower() in langs or language.replace('-', '_') in langs:
                try:
                    logger.info(f"Trying fallback TTS model '{model_name}' for language '{language}'...")
                    tts = TTS(model_name=model_name, progress_bar=False)
                    # Double-check language support
                    if not langs or language in langs or language.lower() in langs or language.replace('-', '_') in langs:
                        logger.info(f"Loaded fallback TTS model '{model_name}' for language '{language}'")
                        return tts
                except Exception as e:
                    logger.error(f"Failed to load fallback TTS model '{model_name}': {e}")
        logger.error(f"No fallback TTS model found that supports language '{language}'")
        return None
    
    def clone_voice(self, text, audio_path, output_path, language=None):
        """
        Clone voice from reference audio and generate speech for given text.
        
        Args:
            text (str): Text to synthesize
            audio_path (str): Path to reference audio file
            output_path (str): Path to save generated audio
            language (str): Language code (optional)
        
        Returns:
            str: Path to generated audio file
        """
        try:
            if self.tts is None:
                raise RuntimeError("TTS model not initialized")
            
            logger.info(f"Cloning voice for text: {text[:50]}...")
            logger.info(f"Reference audio: {audio_path}")
            logger.info(f"Output path: {output_path}")
            logger.info(f"Language: {language}")
            
            # Generate speech with voice cloning
            if language:
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=audio_path,
                    language=language,
                    file_path=output_path
                )
            else:
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=audio_path,
                    file_path=output_path
                )
            
            logger.info(f"Voice cloning completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Voice cloning failed: {e}")
    
    def batch_clone_voice(self, 
                         reference_audio_path: str,
                         texts: List[str],
                         output_dir: str,
                         language: str = "en",
                         speed: float = 1.0) -> List[str]:
        """
        Clone voice for multiple text segments.
        
        Args:
            reference_audio_path: Path to reference audio file
            texts: List of text segments to synthesize
            output_dir: Directory to save generated audio files
            language: Language code
            speed: Speech speed multiplier
            
        Returns:
            List of paths to generated audio files
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        logger.info(f"Starting batch voice cloning for {len(texts)} segments")
        logger.info(f"Reference audio: {reference_audio_path}")
        logger.info(f"Language: {language}, Speed: {speed}")
        
        # Track if we need to use fallback TTS
        use_fallback = False
        fallback_tts = None
        
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"cloned_audio_{i:04d}.wav")
            try:
                logger.info(f"Processing segment {i+1}/{len(texts)}: {text[:50]}...")
                
                if use_fallback and fallback_tts:
                    # Use fallback TTS
                    logger.info(f"Using fallback TTS for segment {i+1}")
                    fallback_tts.tts_to_file(
                        text=text,
                        file_path=output_path
                    )
                else:
                    # Try normal voice cloning
                    self.clone_voice(
                        text=text,
                        audio_path=reference_audio_path,
                        output_path=output_path,
                        language=language
                    )
                
                output_paths.append(output_path)
                logger.info(f"Generated audio {i+1}/{len(texts)}: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate audio {i+1}: {e}")
                logger.error(f"Text: {text}")
                
                # Check if it's the processing frame error
                if "Error while processing frame" in str(e) and not use_fallback:
                    logger.error("Detected processing frame error, switching to fallback TTS")
                    use_fallback = True
                    fallback_tts = self._get_fallback_tts(language)
                    
                    if fallback_tts:
                        try:
                            logger.info(f"Retrying segment {i+1} with fallback TTS")
                            fallback_tts.tts_to_file(
                                text=text,
                                file_path=output_path
                            )
                            output_paths.append(output_path)
                            logger.info(f"Generated fallback audio {i+1}/{len(texts)}: {output_path}")
                            continue
                        except Exception as fallback_error:
                            logger.error(f"Fallback TTS also failed for segment {i+1}: {fallback_error}")
                
                # If we're already using fallback or fallback failed, raise the error
                raise RuntimeError(f"Could not generate audio for segment {i+1}: {e}")
        
        logger.info(f"Batch voice cloning completed. Generated {len(output_paths)} files.")
        return output_paths
    
    def extract_audio_segments(self, 
                             audio_path: str, 
                             segments: List[Tuple[float, float]]) -> List[str]:
        """
        Extract audio segments from a longer audio file.
        
        Args:
            audio_path: Path to audio file
            segments: List of (start_time, end_time) tuples in seconds
            
        Returns:
            List of paths to extracted audio segments
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            extracted_paths = []
            temp_dir = tempfile.mkdtemp()
            
            for i, (start_time, end_time) in enumerate(segments):
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Extract segment
                segment_audio = audio[start_sample:end_sample]
                
                # Save segment
                segment_path = os.path.join(temp_dir, f"segment_{i:04d}.wav")
                sf.write(segment_path, segment_audio, sr)
                extracted_paths.append(segment_path)
            
            return extracted_paths
            
        except Exception as e:
            logger.error(f"Failed to extract audio segments: {e}")
            raise
    
    def process_audio_file(self, 
                          audio_path: str,
                          target_sr: int = 22050,
                          normalize: bool = True) -> str:
        """
        Process audio file for voice cloning (resample, normalize).
        
        Args:
            audio_path: Path to input audio file
            target_sr: Target sample rate
            normalize: Whether to normalize audio
            
        Returns:
            Path to processed audio file
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # Normalize if requested
            if normalize:
                audio = librosa.util.normalize(audio)
            
            # Save processed audio
            temp_dir = tempfile.mkdtemp()
            processed_path = os.path.join(temp_dir, "processed_audio.wav")
            sf.write(processed_path, audio, target_sr)
            
            return processed_path
            
        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            raise
    
    def get_supported_languages(self):
        """Get list of supported languages for the current model."""
        try:
            if self.tts is None:
                return []
            
            # Try to get supported languages from the model
            if hasattr(self.tts, 'languages') and self.tts.languages:
                return self.tts.languages
            else:
                # Default languages for XTTS v2
                return ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'ja', 'ko', 'hi']
        except Exception as e:
            logger.warning(f"Error getting supported languages: {e}")
            return []
    
    def validate_reference_audio(self, audio_path: str) -> bool:
        """
        Validate reference audio for voice cloning.
        
        Args:
            audio_path: Path to reference audio file
            
        Returns:
            True if audio is valid for voice cloning
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"Reference audio file not found: {audio_path}")
                return False
            
            # Load and check audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Check duration (should be at least 3 seconds for good cloning)
            duration = len(audio) / sr
            if duration < 3.0:
                logger.warning(f"Reference audio is too short ({duration:.1f}s). At least 3 seconds recommended.")
            
            # Check for silence
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.01:
                logger.error("Reference audio appears to be silent or too quiet")
                return False
            
            logger.info(f"Reference audio validated: {duration:.1f}s, {sr}Hz")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate reference audio: {e}")
            return False

    def check_snr(self, reference_audio_path, threshold=10):
        """
        Check if the reference audio has sufficient SNR (signal-to-noise ratio).
        Returns True if SNR >= threshold (dB), else False.
        """
        audio, sr = librosa.load(reference_audio_path, sr=22050)
        # Assume first 0.5s is noise, rest is signal
        noise = audio[:int(0.5 * sr)]
        signal = audio[int(0.5 * sr):]
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return True  # No noise
        snr = 10 * np.log10(signal_power / noise_power)
        return snr >= threshold

    def check_vad(self, reference_audio_path, min_speech_percent=0.6):
        """
        Check if the reference audio contains enough speech using VAD.
        Returns True if speech percent >= min_speech_percent, else False.
        """
        audio, sr = sf.read(reference_audio_path)
        if audio.ndim > 1:
            audio = audio[:, 0]  # Use first channel if stereo
        # Convert to 16-bit PCM
        audio_pcm = (audio * 32767).astype(np.int16)
        vad = webrtcvad.Vad(2)
        frame_duration = 30  # ms
        frame_length = int(sr * frame_duration / 1000)
        num_frames = len(audio_pcm) // frame_length
        speech_frames = 0
        for i in range(num_frames):
            start = i * frame_length
            stop = start + frame_length
            frame = audio_pcm[start:stop]
            if len(frame) < frame_length:
                continue
            if vad.is_speech(frame.tobytes(), sr):
                speech_frames += 1
        speech_percent = speech_frames / max(1, num_frames)
        return speech_percent >= min_speech_percent


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