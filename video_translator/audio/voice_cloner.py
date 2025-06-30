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

logger = logging.getLogger(__name__)


class VoiceCloner:
    """Voice cloning using Coqui XTTS for true voice cloning."""
    
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        """
        Initialize the voice cloner with XTTS model.
        
        Args:
            model_name: XTTS model name to use
        """
        self.model_name = model_name
        self.tts = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the XTTS model."""
        try:
            logger.info(f"Loading XTTS model: {self.model_name}")
            
            # Clear memory before loading XTTS
            import gc
            gc.collect()
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fix for PyTorch 2.6+ compatibility
            # Monkey patch torch.load to use weights_only=False for XTTS
            original_torch_load = torch.load
            
            def safe_torch_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            torch.load = safe_torch_load
            
            # Try to initialize with specific device and settings
            try:
                # First try with CPU to avoid GPU issues
                logger.info("Attempting to load XTTS v2 on CPU...")
                self.tts = TTS(model_name=self.model_name, progress_bar=False)
                logger.info("XTTS model loaded successfully on CPU")
            except Exception as cpu_error:
                logger.warning(f"CPU loading failed: {cpu_error}")
                # Try with different model variant
                try:
                    logger.info("Attempting to load XTTS v1.1...")
                    self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v1.1", progress_bar=False)
                    logger.info("XTTS v1.1 model loaded successfully")
                except Exception as v1_error:
                    logger.error(f"XTTS v1.1 also failed: {v1_error}")
                    raise cpu_error
                    
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            logger.info("Falling back to basic TTS model...")
            try:
                # Fallback to a simpler model
                logger.info("Loading fallback TTS model...")
                self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
                logger.info("Fallback TTS model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback TTS model: {fallback_error}")
                # Try one more fallback
                try:
                    logger.info("Trying final fallback TTS model...")
                    self.tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)
                    logger.info("Final fallback TTS model loaded successfully")
                except Exception as final_error:
                    logger.error(f"All TTS models failed to load: {final_error}")
                    raise RuntimeError(f"Could not load any TTS model: {e}")
    
    def _get_fallback_tts(self):
        """Get a fallback TTS model if the main one fails."""
        try:
            return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        except Exception as e:
            logger.error(f"Failed to load fallback TTS: {e}")
            return None
    
    def clone_voice(self, 
                   reference_audio_path: str, 
                   text: str, 
                   output_path: str,
                   language: str = "en",
                   speed: float = 1.0) -> str:
        """
        Clone voice from reference audio and generate speech for given text.
        
        Args:
            reference_audio_path: Path to reference audio file
            text: Text to synthesize
            output_path: Path to save generated audio
            language: Language code (e.g., 'en', 'es', 'fr')
            speed: Speech speed multiplier
            
        Returns:
            Path to generated audio file
        """
        try:
            if self.tts is None:
                raise RuntimeError("TTS model not initialized")
                
            logger.info(f"Cloning voice from {reference_audio_path}")
            logger.info(f"Text to synthesize: {text[:100]}...")
            logger.info(f"Language: {language}, Speed: {speed}")
            
            # Validate inputs
            if not os.path.exists(reference_audio_path):
                raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path}")
            
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Generate speech with voice cloning
            try:
                logger.info("Starting TTS generation...")
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio_path,
                    language=language,
                    file_path=output_path,
                    speed=speed
                )
                logger.info("TTS generation completed successfully")
            except Exception as tts_error:
                logger.error(f"TTS generation failed: {tts_error}")
                logger.error(f"Error type: {type(tts_error).__name__}")
                
                # Check if it's the specific "Error while processing frame" issue
                if "Error while processing frame" in str(tts_error):
                    logger.error("Detected 'Error while processing frame' - this is a known XTTS issue")
                    logger.info("Trying with different parameters...")
                    
                    # Try with different parameters
                    try:
                        self.tts.tts_to_file(
                            text=text,
                            speaker_wav=reference_audio_path,
                            language=language,
                            file_path=output_path
                        )
                        logger.info("TTS generation succeeded with default parameters")
                    except Exception as retry_error:
                        logger.error(f"Retry also failed: {retry_error}")
                        raise RuntimeError(f"XTTS processing frame error: {tts_error}")
                else:
                    # Try with different parameters if the first attempt fails
                    logger.info("Retrying with default speed...")
                    self.tts.tts_to_file(
                        text=text,
                        speaker_wav=reference_audio_path,
                        language=language,
                        file_path=output_path
                    )
            
            # Verify output file was created
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file was not created: {output_path}")
            
            logger.info(f"Voice cloning completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            logger.error(f"Reference audio: {reference_audio_path}")
            logger.error(f"Text length: {len(text)}")
            logger.error(f"Language: {language}")
            raise
    
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
                        reference_audio_path=reference_audio_path,
                        text=text,
                        output_path=output_path,
                        language=language,
                        speed=speed
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
                    fallback_tts = self._get_fallback_tts()
                    
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
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages for XTTS."""
        return [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", 
            "ja", "ko", "hi", "th", "sv", "da", "no", "fi", "hu", "ro", "bg", "hr", "sk", 
            "sl", "et", "lv", "lt", "mt", "el", "he", "id", "ms", "vi", "tl", "ur", "bn", 
            "ta", "te", "ml", "kn", "gu", "pa", "si", "my", "km", "lo", "ne", "am", "or", 
            "as", "sa", "mr", "be", "uk", "ka", "hy", "az", "eu", "gl", "ca", "cy", "ga", 
            "is", "mk", "sq", "bs", "sr", "me", "mn", "ky", "kk", "uz", "tg", "tk", "ps", 
            "fa", "ku", "yi", "bo", "dz", "ug", "wo", "rw", "so", "sw", "zu", "xh", "af", 
            "st", "tn", "ts", "ss", "ve", "ny", "sn", "lg", "rw", "ak", "tw", "ee", "ff", 
            "ha", "ig", "yo", "mg", "om", "ti", "am", "ar", "az", "be", "bg", "bn", "bs", 
            "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", 
            "ga", "gl", "gu", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "ka", 
            "kk", "km", "kn", "ko", "ku", "ky", "lg", "lo", "lt", "lv", "mg", "mk", "ml", 
            "mn", "mr", "ms", "mt", "my", "ne", "nl", "no", "om", "or", "pa", "pl", "ps", 
            "pt", "ro", "rw", "sa", "si", "sk", "sl", "sn", "so", "sq", "sr", "ss", "st", 
            "sv", "sw", "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "tr", "ts", "tw", 
            "ug", "uk", "ur", "uz", "ve", "vi", "wo", "xh", "yi", "yo", "zh-cn", "zu"
        ]
    
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