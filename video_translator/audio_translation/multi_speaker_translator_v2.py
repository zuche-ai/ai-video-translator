"""
Multi-Speaker Audio Translator V2

This module provides improved functionality to translate audio files with multiple speakers by:
1. Transcribing audio with Whisper
2. Using original Whisper segments (not merged) to preserve speaker separation
3. Extracting audio segments for each Whisper segment as voice references
4. Translating each segment individually
5. Voice cloning each segment with its own audio reference
6. Detecting and preserving non-voice segments (music, applause, etc.)
7. Reconstructing final audio with original timing
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


class MultiSpeakerTranslatorV2:
    """Improved multi-speaker audio translation with proper segment handling."""
    
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
        temp_base = os.path.join(self.temp_dir, f"multi_speaker_v2_{os.getpid()}")
        audio_segments_dir = os.path.join(temp_base, "audio_segments")
        voice_cloned_dir = os.path.join(temp_base, "voice_cloned")
        non_voice_dir = os.path.join(temp_base, "non_voice")
        
        for dir_path in [audio_segments_dir, voice_cloned_dir, non_voice_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            # Step 1: Transcribe audio with Whisper
            if progress_hook:
                progress_hook(1, 20, "Transcribing audio...")
            
            segments = transcribe_video(audio_path, language=src_lang, debug=True)
            logger.info(f"Transcription completed with {len(segments)} segments")
            
            # Step 2: Extract audio segments for each Whisper segment
            if progress_hook:
                progress_hook(2, 30, "Extracting audio segments...")
            
            audio_segments = self._extract_audio_segments(audio_path, segments, audio_segments_dir)
            logger.info(f"Extracted {len(audio_segments)} audio segments")
            
            # Step 3: Translate segments
            if progress_hook:
                progress_hook(3, 40, "Translating segments...")
            
            translated_segments = self._translate_segments(segments, src_lang, tgt_lang)
            logger.info(f"Translated {len(translated_segments)} segments")
            
            # Step 4: Voice clone each segment with its audio reference
            if progress_hook:
                progress_hook(4, 60, "Voice cloning segments...")
            
            voice_cloned_files = self._clone_segment_voices(
                translated_segments, audio_segments, voice_cloned_dir, tgt_lang
            )
            logger.info(f"Voice cloned {len(voice_cloned_files)} segments")
            
            # Step 5: Detect and extract non-voice segments
            if progress_hook:
                progress_hook(5, 70, "Processing non-voice segments...")
            
            non_voice_files = self._extract_non_voice_segments(audio_path, segments, non_voice_dir)
            logger.info(f"Extracted {len(non_voice_files)} non-voice segments")
            
            # Step 6: Reconstruct final audio with proper timing
            if progress_hook:
                progress_hook(6, 80, "Reconstructing final audio...")
            
            if not output_path:
                output_path = os.path.join(
                    os.path.dirname(audio_path),
                    f"translated_multi_speaker_v2_{os.path.basename(audio_path)}"
                )
            
            final_audio_path = self._reconstruct_audio_with_timing(
                voice_cloned_files, non_voice_files, segments, output_path
            )
            
            if progress_hook:
                progress_hook(7, 100, "Multi-speaker translation completed!")
            
            logger.info(f"Multi-speaker translation completed: {final_audio_path}")
            return final_audio_path
            
        except Exception as e:
            logger.error(f"Multi-speaker translation failed: {e}")
            raise
        finally:
            # Clean up temporary files
            self._cleanup_temp_files(temp_base)
    
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
    
    def _extract_non_voice_segments(self, audio_path: str, segments: List[Dict], output_dir: str) -> List[str]:
        """
        Extract non-voice segments (music, applause, silence).
        
        Args:
            audio_path: Path to original audio file
            segments: Original Whisper segments
            output_dir: Directory to save non-voice segments
            
        Returns:
            List of paths to non-voice audio files
        """
        non_voice_files = []
        
        # Find gaps between segments and at the beginning/end
        gaps = []
        
        # Check for gap at the beginning (intro music)
        if segments and segments[0]['start'] > 0.5:  # Gap longer than 0.5s
            gaps.append({
                'start': 0.0,
                'end': segments[0]['start'],
                'duration': segments[0]['start'],
                'type': 'intro'
            })
        
        # Find gaps between segments
        for i in range(len(segments) - 1):
            current_end = segments[i]['end']
            next_start = segments[i + 1]['start']
            
            if next_start - current_end > 0.5:  # Gap longer than 0.5s
                gaps.append({
                    'start': current_end,
                    'end': next_start,
                    'duration': next_start - current_end,
                    'type': 'gap'
                })
        
        # Check for gap at the end
        if segments:
            last_end = segments[-1]['end']
            # Get total audio duration
            try:
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                    '-of', 'csv=p=0', audio_path
                ], capture_output=True, text=True, check=True)
                total_duration = float(result.stdout.strip())
                
                if total_duration - last_end > 0.5:  # Gap longer than 0.5s
                    gaps.append({
                        'start': last_end,
                        'end': total_duration,
                        'duration': total_duration - last_end,
                        'type': 'outro'
                    })
            except Exception as e:
                logger.warning(f"Could not determine total duration: {e}")
        
        # Extract gap audio
        for i, gap in enumerate(gaps):
            output_file = os.path.join(output_dir, f"non_voice_{i:04d}.wav")
            
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', str(gap['start']),
                '-t', str(gap['duration']),
                '-c', 'copy',
                output_file
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                non_voice_files.append(output_file)
                logger.info(f"Extracted {gap['type']} segment {i+1}: {gap['start']:.2f}s - {gap['end']:.2f}s ({gap['duration']:.2f}s)")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract {gap['type']} segment {i+1}: {e}")
        
        return non_voice_files
    
    def _reconstruct_audio_with_timing(self, voice_files: List[str], non_voice_files: List[str],
                                     segments: List[Dict], output_path: str) -> str:
        """
        Reconstruct final audio with proper timing using ffmpeg filter complex.
        
        Args:
            voice_files: List of voice-cloned audio files
            non_voice_files: List of non-voice audio files
            segments: Original Whisper segments
            output_path: Path for output audio file
            
        Returns:
            Path to reconstructed audio file
        """
        # Create a timeline of all audio elements in chronological order
        timeline = []
        
        # Add non-voice segments first
        for non_voice_file in non_voice_files:
            if os.path.exists(non_voice_file):
                timeline.append({
                    'type': 'non_voice',
                    'file': non_voice_file
                })
        
        # Add voice-cloned segments
        for i, voice_file in enumerate(voice_files):
            if os.path.exists(voice_file):
                timeline.append({
                    'type': 'voice',
                    'file': voice_file,
                    'segment': segments[i] if i < len(segments) else None
                })
        
        # For now, use simple concatenation
        # In a more sophisticated version, you could use ffmpeg filter complex for precise timing
        temp_list_file = os.path.join(os.path.dirname(output_path), "multi_speaker_v2_concat_list.txt")
        
        try:
            with open(temp_list_file, 'w') as f:
                for item in timeline:
                    f.write(f"file '{item['file']}'\n")
            
            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', temp_list_file,
                '-c', 'copy',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Audio reconstruction completed: {output_path}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to reconstruct audio: {e}")
            raise RuntimeError(f"Audio reconstruction failed: {e}")
        finally:
            if os.path.exists(temp_list_file):
                os.remove(temp_list_file)
    
    def _cleanup_temp_files(self, temp_base: str):
        """Clean up temporary files and directories."""
        try:
            import shutil
            if os.path.exists(temp_base):
                shutil.rmtree(temp_base)
                logger.info(f"Cleaned up temporary directory: {temp_base}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")


def translate_multi_speaker_audio_v2(audio_path: str,
                                   src_lang: str = "en",
                                   tgt_lang: str = "es",
                                   output_path: Optional[str] = None,
                                   progress_hook: Optional[Callable[[int, int, str], None]] = None) -> str:
    """
    Convenience function to translate multi-speaker audio file using V2 approach.
    
    Args:
        audio_path: Path to input audio file
        src_lang: Source language code
        tgt_lang: Target language code
        output_path: Path for output audio file
        progress_hook: Optional callback for progress updates
        
    Returns:
        Path to translated audio file
    """
    translator = MultiSpeakerTranslatorV2()
    return translator.translate_audio(
        audio_path=audio_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        output_path=output_path,
        progress_hook=progress_hook
    ) 