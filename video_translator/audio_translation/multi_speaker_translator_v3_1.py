"""
Multi-Speaker Audio Translator V3.1

This module provides a simple fix to V3 that addresses the voice/music overlap issue:
1. Transcribing audio with Whisper
2. Detecting intro music and non-voice segments
3. Trimming the first voice segment to start after music ends
4. Using only clean voice as reference for voice cloning
5. Reconstructing final audio with proper timing
"""

import os
import tempfile
import logging
import subprocess
import numpy as np
import librosa
from typing import List, Dict, Any, Tuple, Optional, Callable
import re

# Import existing modules
from video_translator.core.transcriber import transcribe_video
from video_translator.core.translator import translate_segments
from video_translator.audio.voice_cloner import VoiceCloner

logger = logging.getLogger(__name__)

class MultiSpeakerTranslatorV3_1:
    """
    Multi-speaker audio translator with simple voice/music separation fix.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir
        self.voice_cloner = VoiceCloner()
        
    def translate_audio(self, 
                       audio_path: str,
                       src_lang: str = "en",
                       tgt_lang: str = "es",
                       output_path: Optional[str] = None,
                       progress_hook: Optional[Callable[[int, int, str], None]] = None) -> str:
        """
        Translate multi-speaker audio with simple voice/music separation.
        
        Args:
            audio_path: Path to input audio file
            src_lang: Source language code
            tgt_lang: Target language code
            output_path: Path for output audio file
            progress_hook: Optional callback for progress updates
            
        Returns:
            Path to translated audio file
        """
        if progress_hook:
            progress_hook(0, 100, "Starting multi-speaker translation...")
            
        # Create temporary directory
        temp_base = tempfile.mkdtemp(prefix="multi_speaker_v3_1_")
        if self.temp_dir:
            temp_base = self.temp_dir
            
        try:
            # Step 1: Analyze audio for non-voice segments
            if progress_hook:
                progress_hook(5, 100, "Analyzing audio for non-voice segments...")
                
            non_voice_segments = self._detect_non_voice_segments(audio_path)
            
            # Step 2: Transcribe audio
            if progress_hook:
                progress_hook(10, 100, "Transcribing audio...")
                
            segments = transcribe_video(audio_path, src_lang)
            logger.info(f"Transcription completed with {len(segments)} segments")
            
            # Step 3: Trim first voice segment if it overlaps with music
            if segments and non_voice_segments:
                segments, first_voice_start = self._trim_first_voice_segment(segments, audio_path)
            
            # Step 4: Extract audio segments
            if progress_hook:
                progress_hook(15, 100, "Extracting audio segments...")
                
            audio_segments = self._extract_audio_segments(audio_path, segments)
            
            # Step 5: Translate segments
            if progress_hook:
                progress_hook(20, 100, "Translating segments...")
                
            translated_segments = translate_segments(segments, src_lang, tgt_lang, debug=True)
            
            # Add translated_text field to match expected format
            for segment in translated_segments:
                segment['translated_text'] = segment['text']
            
            # Step 6: Voice clone segments
            if progress_hook:
                progress_hook(30, 100, "Voice cloning segments...")
                
            voice_cloned_files = self._clone_segment_voices(translated_segments, audio_segments, 
                                                          os.path.join(temp_base, "voice_cloned"), tgt_lang)
            
            # Step 7: Extract non-voice audio
            if progress_hook:
                progress_hook(80, 100, "Extracting non-voice audio...")
                
            non_voice_files = self._extract_non_voice_audio(audio_path, non_voice_segments, 
                                                          os.path.join(temp_base, "non_voice"))
            
            # Step 8: Reconstruct final audio
            if progress_hook:
                progress_hook(90, 100, "Reconstructing final audio...")
                
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = f"translated_multi_speaker_v3_1_{base_name}.wav"
                
            final_audio = self._reconstruct_audio_with_timing(voice_cloned_files, segments, audio_path, first_voice_start, output_path)
            
            if progress_hook:
                progress_hook(100, 100, "Multi-speaker translation completed!")
                
            logger.info(f"Multi-speaker translation completed: {final_audio}")
            return final_audio
            
        finally:
            self._cleanup_temp_files(temp_base)
    
    def _trim_first_voice_segment(self, segments: List[Dict], audio_path: str) -> Tuple[List[Dict], float]:
        """
        Find the true start of speech and return trimmed segments and the speech start time.
        Args:
            segments: Original Whisper segments
            audio_path: Path to audio file
        Returns:
            (trimmed_segments, first_voice_start)
        """
        if not segments:
            return segments, 0.0

        # Find the first segment with real speech (not silence/music)
        first_voice_start = segments[0]['start']
        for seg in segments:
            if seg.get('text', '').strip():
                first_voice_start = seg['start']
                break

        # Trim the first voice segment to start at first_voice_start
        trimmed_segments = []
        for i, seg in enumerate(segments):
            if i == 0 and seg['start'] < first_voice_start:
                seg = seg.copy()
                seg['start'] = first_voice_start
            if seg['end'] > first_voice_start:
                trimmed_segments.append(seg)

        return trimmed_segments, first_voice_start
    
    def _detect_non_voice_segments(self, audio_path: str) -> List[Dict]:
        """
        Detect non-voice segments (intro music, silence, etc.) using audio analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of non-voice segments with timing info
        """
        non_voice_segments = []
        
        # Get total duration
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', audio_path
            ], capture_output=True, text=True, check=True)
            total_duration = float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not determine total duration: {e}")
            return non_voice_segments
        
        # Analyze audio baseline
        baseline_info = self._analyze_audio_baseline(audio_path)
        
        # Detect audio pattern (silence -> music -> speech)
        segments = self._detect_audio_pattern(audio_path, baseline_info, total_duration)

        # --- NEW: Always include silence from 0.0 to first segment if needed ---
        if segments:
            first_start = segments[0]['start']
            if first_start > 0.01:
                non_voice_segments.append({
                    'start': 0.0,
                    'end': first_start,
                    'duration': first_start,
                    'type': 'silence'
                })
        
        non_voice_segments.extend(segments)
        return non_voice_segments
    
    def _analyze_audio_baseline(self, audio_path: str) -> Dict[str, float]:
        """
        Analyze the entire audio to establish adaptive thresholds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with adaptive thresholds
        """
        # Analyze the entire audio for baseline characteristics
        result = subprocess.run([
            'ffmpeg', '-i', audio_path,
            '-af', 'volumedetect',
            '-f', 'null', '/dev/null'
        ], capture_output=True, text=True)
        
        # Parse overall volume levels
        mean_volume_match = re.search(r'mean_volume: ([-\d.]+) dB', result.stderr)
        max_volume_match = re.search(r'max_volume: ([-\d.]+) dB', result.stderr)
        
        if mean_volume_match and max_volume_match:
            mean_volume = float(mean_volume_match.group(1))
            max_volume = float(max_volume_match.group(1))
            
            # Set adaptive thresholds based on overall audio characteristics
            silence_threshold = mean_volume - 10.0  # 10dB below mean
            music_threshold = mean_volume - 5.0     # 5dB below mean
            speech_threshold = mean_volume + 5.0    # 5dB above mean
            
            logger.info(f"Audio baseline: silence_threshold={silence_threshold:.1f}dB, music_threshold={music_threshold:.1f}dB, speech_threshold={speech_threshold:.1f}dB")
            
            return {
                'mean_volume': mean_volume,
                'max_volume': max_volume,
                'silence_threshold': silence_threshold,
                'music_threshold': music_threshold,
                'speech_threshold': speech_threshold
            }
        else:
            # Fallback thresholds
            logger.warning("Could not parse volume levels, using fallback thresholds")
            return {
                'mean_volume': -30.0,
                'max_volume': -10.0,
                'silence_threshold': -40.0,
                'music_threshold': -35.0,
                'speech_threshold': -25.0
            }
    
    def _detect_audio_pattern(self, audio_path: str, baseline_info: Dict[str, float], total_duration: float) -> List[Dict]:
        """
        Detect the pattern: silence -> music -> speech.
        
        Args:
            audio_path: Path to audio file
            baseline_info: Audio baseline information
            total_duration: Total audio duration
            
        Returns:
            List of non-voice segments
        """
        segments = []
        silence_threshold = baseline_info['silence_threshold']
        music_threshold = baseline_info['music_threshold']
        
        # Analyze first 30 seconds in detail
        search_duration = min(30.0, total_duration)
        window_size = 2.0
        step_size = 1.0
        
        current_state = 'silence'
        silence_start = 0.0
        music_start = None
        speech_start = None
        
        for start_time in np.arange(0, search_duration - window_size, step_size):
            mean_volume = self._analyze_window_volume(audio_path, start_time, window_size)
            
            if current_state == 'silence':
                if mean_volume > silence_threshold:
                    # Transition from silence to music
                    current_state = 'music'
                    music_start = start_time
                    logger.info(f"Music detected starting at {music_start:.2f}s (volume: {mean_volume:.1f}dB)")
            elif current_state == 'music':
                if mean_volume > music_threshold + 10:  # Significant increase
                    # Transition from music to speech
                    current_state = 'speech'
                    speech_start = start_time
                    logger.info(f"Speech detected starting at {speech_start:.2f}s (volume: {mean_volume:.1f}dB)")
                    break
        
        # Create segments based on detected pattern
        if current_state == 'speech':
            # We found the complete pattern
            if music_start is not None:
                # Had both silence and music
                if music_start > 0.5:  # Only add silence if it's substantial
                    segments.append({
                        'start': 0.0,
                        'end': music_start,
                        'duration': music_start,
                        'type': 'silence'
                    })
                
                segments.append({
                    'start': music_start,
                    'end': speech_start,
                    'duration': speech_start - music_start,
                    'type': 'music'
                })
            else:
                # Had silence, then speech directly (no music)
                if speech_start is not None and speech_start > 0.5:  # Only add silence if it's substantial
                    segments.append({
                        'start': 0.0,
                        'end': speech_start,
                        'duration': speech_start,
                        'type': 'silence'
                    })
        else:
            # Incomplete pattern, add what we found
            if music_start is not None and music_start > 0.0:
                segments.append({
                    'start': music_start,
                    'end': search_duration,
                    'duration': search_duration - music_start,
                    'type': 'music'
                })
        
        # Log detected segments
        for segment in segments:
            logger.info(f"Detected {segment['type']}: {segment['start']:.2f}s - {segment['end']:.2f}s ({segment['duration']:.2f}s)")
        
        logger.info(f"Detected {len(segments)} non-voice segments")
        return segments
    
    def _analyze_window_volume(self, audio_path: str, start_time: float, window_size: float) -> float:
        """
        Analyze volume in a specific time window.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time of window
            window_size: Duration of window
            
        Returns:
            Mean volume in dB
        """
        # Extract the window
        temp_window = f"/tmp/volume_analysis_{start_time:.2f}.wav"
        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', str(start_time),
            '-t', str(window_size),
            '-c', 'copy',
            temp_window
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Analyze volume
            result = subprocess.run([
                'ffmpeg', '-i', temp_window,
                '-af', 'volumedetect',
                '-f', 'null', '/dev/null'
            ], capture_output=True, text=True)
            
            # Parse mean volume
            mean_volume_match = re.search(r'mean_volume: ([-\d.]+) dB', result.stderr)
            if mean_volume_match:
                return float(mean_volume_match.group(1))
            else:
                return -60.0  # Very quiet if we can't detect
                
        except Exception as e:
            logger.warning(f"Failed to analyze window at {start_time:.2f}s: {e}")
            return -60.0
        finally:
            if os.path.exists(temp_window):
                os.remove(temp_window)
    
    def _extract_audio_segments(self, audio_path: str, segments: List[Dict]) -> List[str]:
        """
        Extract audio segments for voice cloning.
        
        Args:
            audio_path: Path to original audio file
            segments: List of segments with timing info
            
        Returns:
            List of paths to audio segments
        """
        audio_segments = []
        output_dir = os.path.join(os.path.dirname(audio_path), "temp_audio_segments")
        os.makedirs(output_dir, exist_ok=True)
        
        for i, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            
            if duration > 0:
                output_file = os.path.join(output_dir, f"segment_{i:04d}.wav")
                
                cmd = [
                    'ffmpeg', '-y', '-i', audio_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    output_file
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    audio_segments.append(output_file)
                    logger.info(f"Extracted segment {i+1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
                    logger.info(f"  Text: {segment['text'][:50]}...")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to extract segment {i+1}: {e}")
                    audio_segments.append("")
            else:
                logger.warning(f"Segment {i+1} has zero duration")
                audio_segments.append("")
        
        logger.info(f"Extracted {len([s for s in audio_segments if s])} audio segments")
        return audio_segments
    
    def _clone_segment_voices(self, segments: List[Dict], audio_segments: List[str], 
                            output_dir: str, language: str) -> List[str]:
        """
        Voice clone each segment using its audio reference.
        
        Args:
            segments: List of segments with translated text
            audio_segments: List of audio segment paths
            output_dir: Directory to save voice-cloned files
            language: Target language for voice cloning
            
        Returns:
            List of paths to voice-cloned audio files
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        voice_cloned_files = []
        
        for i, segment in enumerate(segments):
            if i < len(audio_segments) and os.path.exists(audio_segments[i]):
                translated_text = segment['translated_text']
                reference_audio = audio_segments[i]
                
                output_file = os.path.join(output_dir, f"cloned_{i:04d}.wav")
                
                try:
                    # Voice clone this segment
                    self.voice_cloner.clone_voice(
                        text=translated_text,
                        audio_path=reference_audio,
                        output_path=output_file,
                        language=language
                    )
                    
                    voice_cloned_files.append(output_file)
                    logger.info(f"Voice cloned segment {i+1}: {translated_text[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Failed to voice clone segment {i+1}: {e}")
                    voice_cloned_files.append("")
            else:
                logger.warning(f"No audio reference for segment {i+1}")
                voice_cloned_files.append("")
        
        return voice_cloned_files
    
    def _extract_non_voice_audio(self, audio_path: str, non_voice_segments: List[Dict], output_dir: str) -> List[str]:
        """
        Extract non-voice audio segments.
        
        Args:
            audio_path: Path to original audio file
            non_voice_segments: List of non-voice segments
            output_dir: Directory to save non-voice segments
            
        Returns:
            List of paths to non-voice audio files
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        non_voice_files = []
        
        for i, segment in enumerate(non_voice_segments):
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            
            output_file = os.path.join(output_dir, f"non_voice_{i:04d}.wav")
            
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',
                output_file
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                non_voice_files.append(output_file)
                logger.info(f"Extracted {segment['type']} segment {i+1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract {segment['type']} segment {i+1}: {e}")
        
        return non_voice_files
    
    def _reconstruct_audio_with_timing(self, voice_files: List[str], segments: List[Dict], audio_path: str, first_voice_start: float, output_path: str) -> str:
        """
        Reconstruct final audio: [original intro (0.0 to first_voice_start)] + [voice segments].
        Args:
            voice_files: List of voice-cloned audio files
            segments: Whisper segments (speech only)
            audio_path: Path to original audio file
            first_voice_start: Time where speech starts
            output_path: Path for output audio file
        Returns:
            Path to reconstructed audio file
        """
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp(prefix="recon_v3_1_")
        temp_files = []
        try:
            # Step 1: Extract original intro (0.0 to first_voice_start) in original format
            intro_file = os.path.join(temp_dir, "intro.wav")
            if first_voice_start > 0.01:
                cmd = [
                    'ffmpeg', '-y', '-i', audio_path,
                    '-ss', '0.0', '-t', str(first_voice_start),
                    '-c', 'copy',  # Preserve original format exactly
                    intro_file
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                temp_files.append(intro_file)
            # Step 2: Convert voice segments to original format to match intro
            voice_files_original = []
            for voice_file in voice_files:
                if os.path.exists(voice_file):
                    voice_original = os.path.join(temp_dir, f"voice_original_{len(voice_files_original)}.wav")
                    cmd = [
                        'ffmpeg', '-y', '-i', voice_file,
                        '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
                        voice_original
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    voice_files_original.append(voice_original)
            
            # Step 3: Add all voice segments (converted to original format)
            temp_files.extend(voice_files_original)
            
            # Step 4: Concatenate all segments in original format
            temp_list_file = os.path.join(temp_dir, "concat_list.txt")
            with open(temp_list_file, 'w') as f:
                for file_path in temp_files:
                    f.write(f"file '{file_path}'\n")
            
            # Step 5: Concatenate in original format first
            temp_concat = os.path.join(temp_dir, "temp_concat.wav")
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', temp_list_file,
                temp_concat
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Step 6: Keep in original format (no resampling to avoid artifacts)
            cmd = [
                'ffmpeg', '-y', '-i', temp_concat,
                '-c', 'copy',  # Preserve original format exactly
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        finally:
            shutil.rmtree(temp_dir)
    
    def _cleanup_temp_files(self, temp_base: str):
        """Clean up temporary files and directories."""
        try:
            import shutil
            if os.path.exists(temp_base):
                shutil.rmtree(temp_base)
                logger.info(f"Cleaned up temporary directory: {temp_base}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")


def translate_multi_speaker_audio_v3_1(audio_path: str,
                                     src_lang: str = "en",
                                     tgt_lang: str = "es",
                                     output_path: Optional[str] = None,
                                     progress_hook: Optional[Callable[[int, int, str], None]] = None) -> str:
    """
    Convenience function to translate multi-speaker audio file using V3.1 approach.
    
    Args:
        audio_path: Path to input audio file
        src_lang: Source language code
        tgt_lang: Target language code
        output_path: Path for output audio file
        progress_hook: Optional callback for progress updates
        
    Returns:
        Path to translated audio file
    """
    translator = MultiSpeakerTranslatorV3_1()
    return translator.translate_audio(
        audio_path=audio_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        output_path=output_path,
        progress_hook=progress_hook
    ) 