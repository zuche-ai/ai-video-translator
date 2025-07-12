"""
Multi-Speaker Audio Translator V3

This module provides improved functionality to translate audio files with multiple speakers by:
1. Transcribing audio with Whisper
2. Using original Whisper segments to preserve speaker separation
3. Detecting intro music and non-voice segments using audio analysis
4. Extracting audio segments for each Whisper segment as voice references
5. Translating each segment individually
6. Voice cloning each segment with its own audio reference
7. Reconstructing final audio with proper timing including intro music
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


class MultiSpeakerTranslatorV3:
    """Improved multi-speaker audio translation with intro music detection."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the multi-speaker translator.
        
        Args:
            temp_dir: Directory for temporary files (defaults to system temp dir)
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.voice_cloner = VoiceCloner()
    
    def translate_audio(self, 
                       audio_path: str,
                       src_lang: str = "en",
                       tgt_lang: str = "es",
                       output_path: Optional[str] = None,
                       progress_hook: Optional[Callable[[int, int, str], None]] = None) -> str:
        """
        Main method to translate multi-speaker audio file.
        
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
            progress_hook(0, 10, "Starting multi-speaker translation...")
        
        # Create temporary directories
        temp_base = os.path.join(self.temp_dir, f"multi_speaker_v3_{os.getpid()}")
        audio_segments_dir = os.path.join(temp_base, "audio_segments")
        voice_cloned_dir = os.path.join(temp_base, "voice_cloned")
        non_voice_dir = os.path.join(temp_base, "non_voice")
        
        for dir_path in [audio_segments_dir, voice_cloned_dir, non_voice_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            # Step 1: Detect intro music and non-voice segments
            if progress_hook:
                progress_hook(1, 15, "Analyzing audio for non-voice segments...")
            
            non_voice_segments = self._detect_non_voice_segments(audio_path)
            logger.info(f"Detected {len(non_voice_segments)} non-voice segments")
            
            # Step 2: Transcribe audio with Whisper
            if progress_hook:
                progress_hook(2, 25, "Transcribing audio...")
            
            segments = transcribe_video(audio_path, language=src_lang, debug=True)
            logger.info(f"Transcription completed with {len(segments)} segments")
            
            # Step 3: Extract audio segments for each Whisper segment
            if progress_hook:
                progress_hook(3, 35, "Extracting audio segments...")
            
            audio_segments = self._extract_audio_segments(audio_path, segments, audio_segments_dir)
            logger.info(f"Extracted {len(audio_segments)} audio segments")
            
            # Step 4: Translate segments
            if progress_hook:
                progress_hook(4, 45, "Translating segments...")
            
            translated_segments = self._translate_segments(segments, src_lang, tgt_lang)
            logger.info(f"Translated {len(translated_segments)} segments")
            
            # Step 5: Voice clone each segment with its audio reference
            if progress_hook:
                progress_hook(5, 65, "Voice cloning segments...")
            
            voice_cloned_files = self._clone_segment_voices(
                translated_segments, audio_segments, voice_cloned_dir, tgt_lang
            )
            logger.info(f"Voice cloned {len(voice_cloned_files)} segments")
            
            # Step 6: Extract non-voice audio segments
            if progress_hook:
                progress_hook(6, 75, "Extracting non-voice audio...")
            
            non_voice_files = self._extract_non_voice_audio(audio_path, non_voice_segments, non_voice_dir)
            logger.info(f"Extracted {len(non_voice_files)} non-voice audio files")
            
            # Step 7: Reconstruct final audio with proper timing
            if progress_hook:
                progress_hook(7, 85, "Reconstructing final audio...")
            
            if not output_path:
                output_path = os.path.join(
                    os.path.dirname(audio_path),
                    f"translated_multi_speaker_v3_{os.path.basename(audio_path)}"
                )
            
            final_audio_path = self._reconstruct_audio_with_timing(
                voice_cloned_files, non_voice_files, segments, non_voice_segments, output_path
            )
            
            if progress_hook:
                progress_hook(8, 100, "Multi-speaker translation completed!")
            
            logger.info(f"Multi-speaker translation completed: {final_audio_path}")
            return final_audio_path
            
        except Exception as e:
            logger.error(f"Multi-speaker translation failed: {e}")
            raise
        finally:
            # Clean up temporary files
            self._cleanup_temp_files(temp_base)
    
    def _detect_non_voice_segments(self, audio_path: str) -> List[Dict]:
        """
        Detect non-voice segments (silence, intro music, etc.) using adaptive audio analysis.
        
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
        
        # Analyze the entire audio to establish baseline characteristics
        baseline_info = self._analyze_audio_baseline(audio_path)
        logger.info(f"Audio baseline: silence_threshold={baseline_info['silence_threshold']:.1f}dB, "
                   f"music_threshold={baseline_info['music_threshold']:.1f}dB, "
                   f"speech_threshold={baseline_info['speech_threshold']:.1f}dB")
        
        # Search for the pattern: Silence → Music → Speech
        pattern_segments = self._detect_audio_pattern(audio_path, baseline_info, total_duration)
        
        # Add detected non-voice segments
        for segment in pattern_segments:
            if segment['type'] in ['silence', 'music']:
                non_voice_segments.append(segment)
                logger.info(f"Detected {segment['type']}: {segment['start']:.2f}s - {segment['end']:.2f}s "
                           f"({segment['duration']:.2f}s)")
        
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
        
        if not mean_volume_match or not max_volume_match:
            # Fallback to default thresholds
            return {
                'silence_threshold': -45.0,
                'music_threshold': -35.0,
                'speech_threshold': -25.0
            }
        
        overall_mean = float(mean_volume_match.group(1))
        overall_max = float(max_volume_match.group(1))
        
        # Calculate adaptive thresholds based on the audio characteristics
        # Silence: Very low volume (near the minimum)
        silence_threshold = overall_mean - 15.0
        
        # Music: Moderate volume (between silence and speech)
        music_threshold = overall_mean - 5.0
        
        # Speech: Higher volume (closer to the overall mean)
        speech_threshold = overall_mean + 5.0
        
        # Ensure reasonable bounds
        silence_threshold = max(-60.0, min(-30.0, silence_threshold))
        music_threshold = max(-50.0, min(-20.0, music_threshold))
        speech_threshold = max(-40.0, min(-10.0, speech_threshold))
        
        return {
            'silence_threshold': silence_threshold,
            'music_threshold': music_threshold,
            'speech_threshold': speech_threshold,
            'overall_mean': overall_mean,
            'overall_max': overall_max
        }
    
    def _detect_audio_pattern(self, audio_path: str, baseline_info: Dict[str, float], total_duration: float) -> List[Dict]:
        """
        Detect the pattern: Silence → Music → Speech in the audio.
        
        Args:
            audio_path: Path to audio file
            baseline_info: Adaptive thresholds
            total_duration: Total audio duration
            
        Returns:
            List of detected segments with timing info
        """
        segments = []
        window_size = 2.0  # 2-second analysis windows
        step_size = 0.5    # 0.5-second steps
        
        silence_threshold = baseline_info['silence_threshold']
        music_threshold = baseline_info['music_threshold']
        speech_threshold = baseline_info['speech_threshold']
        
        # Search for the pattern in the first 30 seconds (or total duration if shorter)
        search_duration = min(30.0, total_duration)
        
        current_state = 'searching'
        silence_start = 0.0
        music_start = 0.0
        speech_start = 0.0
        
        for start_time in np.arange(0, search_duration - window_size, step_size):
            # Analyze current window
            window_mean = self._analyze_window_volume(audio_path, start_time, window_size)
            
            if current_state == 'searching':
                # Looking for the start of audio activity
                if window_mean > silence_threshold:
                    if window_mean <= music_threshold:
                        # Found music
                        music_start = start_time
                        current_state = 'in_music'
                        logger.info(f"Music detected starting at {start_time:.2f}s (volume: {window_mean:.1f}dB)")
                    elif window_mean > speech_threshold:
                        # Found speech directly (no music)
                        speech_start = start_time
                        current_state = 'in_speech'
                        logger.info(f"Speech detected starting at {start_time:.2f}s (volume: {window_mean:.1f}dB)")
                        break
                    else:
                        # Found some audio activity, but not clearly music or speech
                        # Continue searching
                        pass
            
            elif current_state == 'in_music':
                # Currently in music, looking for transition to speech
                if window_mean > speech_threshold:
                    # Music ended, speech started
                    speech_start = start_time
                    current_state = 'in_speech'
                    logger.info(f"Speech detected starting at {start_time:.2f}s (volume: {window_mean:.1f}dB)")
                    break
        
        # Create segments based on detected pattern
        if current_state == 'in_speech':
            # We found the complete pattern
            if music_start > 0:
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
                if speech_start > 0.5:  # Only add silence if it's substantial
                    segments.append({
                        'start': 0.0,
                        'end': speech_start,
                        'duration': speech_start,
                        'type': 'silence'
                    })
        elif current_state == 'in_music':
            # Found music but didn't reach speech yet
            if music_start > 0:
                if music_start > 0.5:  # Only add silence if it's substantial
                    segments.append({
                        'start': 0.0,
                        'end': music_start,
                        'duration': music_start,
                        'type': 'silence'
                    })
                
                segments.append({
                    'start': music_start,
                    'end': search_duration,
                    'duration': search_duration - music_start,
                    'type': 'music'
                })
        
        return segments
    
    def _analyze_window_volume(self, audio_path: str, start_time: float, window_size: float) -> float:
        """
        Analyze the mean volume of a specific time window.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time of window
            window_size: Duration of window
            
        Returns:
            Mean volume in dB
        """
        temp_window = os.path.join(self.temp_dir, f"volume_window_{os.getpid()}.wav")
        
        try:
            # Extract window
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(window_size),
                '-c', 'copy',
                temp_window
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Analyze volume
            result = subprocess.run([
                'ffmpeg', '-i', temp_window,
                '-af', 'volumedetect',
                '-f', 'null', '/dev/null'
            ], capture_output=True, text=True)
            
            mean_volume_match = re.search(r'mean_volume: ([-\d.]+) dB', result.stderr)
            
            if mean_volume_match:
                return float(mean_volume_match.group(1))
            else:
                return -60.0  # Very quiet if we can't detect
            
        except Exception as e:
            logger.warning(f"Failed to analyze window at {start_time}s: {e}")
            return -60.0
        finally:
            if os.path.exists(temp_window):
                os.remove(temp_window)
    
    def _find_speech_start(self, audio_path: str, max_search_duration: float) -> float:
        """
        Find where speech actually starts in the audio.
        
        Args:
            audio_path: Path to audio file
            max_search_duration: Maximum duration to search
            
        Returns:
            Time in seconds where speech starts
        """
        # Use a sliding window approach to find speech
        window_size = 2.0  # 2-second windows
        step_size = 0.5    # 0.5-second steps
        
        for start_time in np.arange(0, max_search_duration - window_size, step_size):
            end_time = start_time + window_size
            
            # Extract window
            temp_window = os.path.join(self.temp_dir, f"speech_window_{os.getpid()}.wav")
            
            try:
                cmd = [
                    'ffmpeg', '-y', '-i', audio_path,
                    '-ss', str(start_time),
                    '-t', str(window_size),
                    '-c', 'copy',
                    temp_window
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Analyze volume
                result = subprocess.run([
                    'ffmpeg', '-i', temp_window,
                    '-af', 'volumedetect',
                    '-f', 'null', '/dev/null'
                ], capture_output=True, text=True)
                
                mean_volume_match = re.search(r'mean_volume: ([-\d.]+) dB', result.stderr)
                
                if mean_volume_match:
                    mean_volume = float(mean_volume_match.group(1))
                    
                    # If volume is high enough, speech has started
                    if mean_volume > -25.0:
                        logger.info(f"Speech detected starting at {start_time:.2f}s (volume: {mean_volume}dB)")
                        return start_time
                
            except Exception as e:
                logger.warning(f"Failed to analyze window at {start_time}s: {e}")
            finally:
                if os.path.exists(temp_window):
                    os.remove(temp_window)
        
        return 0.0  # No clear speech start found
    
    def _extract_audio_segments(self, audio_path: str, segments: List[Dict], output_dir: str) -> List[str]:
        """
        Extract audio segments for each Whisper segment.
        
        Args:
            audio_path: Path to original audio file
            segments: List of Whisper segments
            output_dir: Directory to save audio segments
            
        Returns:
            List of paths to extracted audio segments
        """
        audio_segments = []
        
        for i, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            
            output_file = os.path.join(output_dir, f"segment_{i:04d}.wav")
            
            # Use ffmpeg to extract segment
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
                logger.info(f"  Text: {segment['text'][:100]}...")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract segment {i+1}: {e}")
        
        return audio_segments
    
    def _translate_segments(self, segments: List[Dict], src_lang: str, tgt_lang: str) -> List[Dict]:
        """
        Translate segments.
        
        Args:
            segments: List of Whisper segments
            src_lang: Source language
            tgt_lang: Target language
            
        Returns:
            List of segments with translated text
        """
        # Translate
        translated_segments = translate_segments(segments, src_lang, tgt_lang, debug=True)
        
        # Add translated text to original segments
        for i, segment in enumerate(segments):
            segment['translated_text'] = translated_segments[i]['text']
        
        return segments
    
    def _clone_segment_voices(self, segments: List[Dict], audio_segments: List[str], 
                            output_dir: str, language: str) -> List[str]:
        """
        Voice clone each segment with its audio reference.
        
        Args:
            segments: List of segments with translated text
            audio_segments: List of audio segment files
            output_dir: Directory to save voice-cloned files
            language: Target language
            
        Returns:
            List of paths to voice-cloned audio files
        """
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
    
    def _reconstruct_audio_with_timing(self, voice_files: List[str], non_voice_files: List[str],
                                     segments: List[Dict], non_voice_segments: List[Dict], output_path: str) -> str:
        """
        Reconstruct final audio with proper timing: non-voice segments first, then voice segments with no overlap.
        Args:
            voice_files: List of voice-cloned audio files
            non_voice_files: List of non-voice audio files
            segments: Original Whisper segments
            non_voice_segments: Non-voice segments
            output_path: Path for output audio file
        Returns:
            Path to reconstructed audio file
        """
        # Set target sample rate and format
        target_sr = 24000
        target_ac = 1
        target_fmt = 's16'
        temp_dir = os.path.dirname(output_path)
        temp_resampled = []
        final_timeline = []

        # Step 1: Process non-voice segments first (silence, music)
        for i, nv_segment in enumerate(non_voice_segments):
            if i < len(non_voice_files) and os.path.exists(non_voice_files[i]):
                start_time = nv_segment['start']
                end_time = nv_segment['end']
                
                # For music segments, trim to end before first voice starts
                if nv_segment['type'] == 'music' and segments:
                    first_voice_start = segments[0]['start']
                    if first_voice_start > start_time:
                        end_time = min(end_time, first_voice_start)
                
                duration = end_time - start_time
                if duration > 0:
                    # Extract the non-voice segment
                    extracted_file = os.path.join(temp_dir, f"extracted_{nv_segment['type']}_{i:04d}.wav")
                    cmd = [
                        'ffmpeg', '-y', '-i', non_voice_files[i],
                        '-t', str(duration),
                        '-ar', str(target_sr), '-ac', str(target_ac), '-sample_fmt', target_fmt,
                        extracted_file
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    temp_resampled.append(extracted_file)
                    final_timeline.append(extracted_file)
                    logger.info(f"Added {nv_segment['type']} segment: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")

        # Step 2: Process voice segments (only the voice parts, no music underneath)
        for i, segment in enumerate(segments):
            if i < len(voice_files) and os.path.exists(voice_files[i]):
                # Resample voice segment to target format
                resampled_voice = os.path.join(temp_dir, f"resampled_voice_{i:04d}.wav")
                cmd = [
                    'ffmpeg', '-y', '-i', voice_files[i],
                    '-ar', str(target_sr), '-ac', str(target_ac), '-sample_fmt', target_fmt,
                    resampled_voice
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                temp_resampled.append(resampled_voice)
                final_timeline.append(resampled_voice)
                logger.info(f"Added voice segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")

        # Step 3: Create concat file
        temp_list_file = os.path.join(temp_dir, "multi_speaker_v3_concat_list.txt")
        with open(temp_list_file, 'w') as f:
            for file_path in final_timeline:
                f.write(f"file '{file_path}'\n")

        # Step 4: Use ffmpeg to concatenate
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list_file,
            '-ar', str(target_sr), '-ac', str(target_ac), '-sample_fmt', target_fmt,
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Audio reconstruction completed: {output_path}")

        # Clean up temp files
        for fpath in temp_resampled:
            if os.path.exists(fpath):
                os.remove(fpath)
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)

        return output_path
    
    def _cleanup_temp_files(self, temp_base: str):
        """Clean up temporary files and directories."""
        try:
            import shutil
            if os.path.exists(temp_base):
                shutil.rmtree(temp_base)
                logger.info(f"Cleaned up temporary directory: {temp_base}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")


def translate_multi_speaker_audio_v3(audio_path: str,
                                   src_lang: str = "en",
                                   tgt_lang: str = "es",
                                   output_path: Optional[str] = None,
                                   progress_hook: Optional[Callable[[int, int, str], None]] = None) -> str:
    """
    Convenience function to translate multi-speaker audio file using V3 approach.
    
    Args:
        audio_path: Path to input audio file
        src_lang: Source language code
        tgt_lang: Target language code
        output_path: Path for output audio file
        progress_hook: Optional callback for progress updates
        
    Returns:
        Path to translated audio file
    """
    translator = MultiSpeakerTranslatorV3()
    return translator.translate_audio(
        audio_path=audio_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        output_path=output_path,
        progress_hook=progress_hook
    ) 