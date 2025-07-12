"""
Multi-Speaker Audio Translator

This module provides functionality to translate audio files with multiple speakers by:
1. Transcribing audio with Whisper
2. Merging fragmented segments into complete sentences
3. Extracting corresponding audio segments as voice references
4. Translating complete sentences
5. Voice cloning each sentence with its own audio reference
6. Handling non-voice segments (music, applause, etc.)
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


class MultiSpeakerTranslator:
    """Multi-speaker audio translation with sentence merging and voice cloning."""
    
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
        temp_base = os.path.join(self.temp_dir, f"multi_speaker_translate_{os.getpid()}")
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
            
            # Step 2: Merge segments into complete sentences
            if progress_hook:
                progress_hook(2, 30, "Merging segments into sentences...")
            
            merged_sentences = self._merge_segments_into_sentences(segments)
            logger.info(f"Merged {len(segments)} segments into {len(merged_sentences)} sentences")
            
            # Step 3: Extract audio segments for each sentence
            if progress_hook:
                progress_hook(3, 40, "Extracting audio segments...")
            
            audio_segments = self._extract_audio_segments(audio_path, merged_sentences, audio_segments_dir)
            logger.info(f"Extracted {len(audio_segments)} audio segments")
            
            # Step 4: Translate sentences
            if progress_hook:
                progress_hook(4, 50, "Translating sentences...")
            
            translated_sentences = self._translate_sentences(merged_sentences, src_lang, tgt_lang)
            logger.info(f"Translated {len(translated_sentences)} sentences")
            
            # Step 5: Voice clone each sentence with its audio reference
            if progress_hook:
                progress_hook(5, 70, "Voice cloning sentences...")
            
            voice_cloned_files = self._clone_sentence_voices(
                translated_sentences, audio_segments, voice_cloned_dir, tgt_lang
            )
            logger.info(f"Voice cloned {len(voice_cloned_files)} sentences")
            
            # Step 6: Handle non-voice segments
            if progress_hook:
                progress_hook(6, 80, "Processing non-voice segments...")
            
            non_voice_files = self._extract_non_voice_segments(audio_path, segments, non_voice_dir)
            logger.info(f"Extracted {len(non_voice_files)} non-voice segments")
            
            # Step 7: Reconstruct final audio
            if progress_hook:
                progress_hook(7, 90, "Reconstructing final audio...")
            
            if not output_path:
                output_path = os.path.join(
                    os.path.dirname(audio_path),
                    f"translated_multi_speaker_{os.path.basename(audio_path)}"
                )
            
            final_audio_path = self._reconstruct_audio(
                voice_cloned_files, non_voice_files, merged_sentences, segments, output_path
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
    
    def _merge_segments_into_sentences(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge Whisper segments into complete sentences.
        
        Args:
            segments: List of Whisper segments
            
        Returns:
            List of merged sentences with timing info
        """
        if not segments:
            return []
        
        # Combine all text from segments
        all_text = " ".join([seg['text'].strip() for seg in segments if seg['text'].strip()])
        logger.info(f"Combined text length: {len(all_text)} characters")
        
        # Handle abbreviations and initials
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.']
        text_with_markers = all_text
        for abbr in abbreviations:
            text_with_markers = text_with_markers.replace(abbr, abbr.replace('.', '###PERIOD###'))
        
        # Replace single letter initials
        text_with_markers = re.sub(r'\b([A-Z])\.(?=\s|$)', r'\1###PERIOD###', text_with_markers)
        
        # Split into sentences and preserve punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text_with_markers)
        sentences = [s.replace('###PERIOD###', '.') for s in sentences]
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Split long sentences (over 239 characters for XTTS)
        MAX_CHAR_LENGTH = 239
        final_sentences = []
        
        for sentence in sentences:
            if len(sentence) <= MAX_CHAR_LENGTH:
                final_sentences.append(sentence)
            else:
                # Split at commas first, then spaces
                if ',' in sentence:
                    parts = sentence.split(',')
                    current_part = parts[0]
                    
                    for part in parts[1:]:
                        if len(current_part + ',' + part) <= MAX_CHAR_LENGTH:
                            current_part += ',' + part
                        else:
                            if current_part.strip():
                                final_sentences.append(current_part.strip())
                            current_part = part
                    
                    if current_part.strip():
                        final_sentences.append(current_part.strip())
                else:
                    # Split at spaces
                    words = sentence.split()
                    current_part = words[0]
                    
                    for word in words[1:]:
                        if len(current_part + ' ' + word) <= MAX_CHAR_LENGTH:
                            current_part += ' ' + word
                        else:
                            if current_part.strip():
                                final_sentences.append(current_part.strip())
                            current_part = word
                    
                    if current_part.strip():
                        final_sentences.append(current_part.strip())
        
        # Map sentences back to timing information
        merged_sentences = []
        sentence_index = 0
        
        for sentence in final_sentences:
            # Find which original segments this sentence came from
            sentence_words = set(sentence.lower().split())
            matching_segments = []
            
            for seg in segments:
                seg_words = set(seg['text'].lower().split())
                if sentence_words.intersection(seg_words):
                    matching_segments.append(seg)
            
            if matching_segments:
                start_time = min(seg['start'] for seg in matching_segments)
                end_time = max(seg['end'] for seg in matching_segments)
                
                merged_sentences.append({
                    'index': sentence_index,
                    'text': sentence,
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'segments': matching_segments
                })
                sentence_index += 1
        
        logger.info(f"Created {len(merged_sentences)} merged sentences:")
        for i, sentence in enumerate(merged_sentences):
            logger.info(f"  Sentence {i+1}: {sentence['start']:.2f}s - {sentence['end']:.2f}s")
            logger.info(f"    Text: {sentence['text'][:100]}...")
        
        return merged_sentences
    
    def _extract_audio_segments(self, audio_path: str, sentences: List[Dict], output_dir: str) -> List[str]:
        """
        Extract audio segments for each sentence.
        
        Args:
            audio_path: Path to original audio file
            sentences: List of sentences with timing info
            output_dir: Directory to save audio segments
            
        Returns:
            List of paths to extracted audio segments
        """
        audio_segments = []
        
        for i, sentence in enumerate(sentences):
            start_time = sentence['start']
            end_time = sentence['end']
            duration = end_time - start_time
            
            output_file = os.path.join(output_dir, f"sentence_{i:04d}.wav")
            
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
                logger.info(f"Extracted sentence {i+1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract sentence {i+1}: {e}")
        
        return audio_segments
    
    def _translate_sentences(self, sentences: List[Dict], src_lang: str, tgt_lang: str) -> List[Dict]:
        """
        Translate sentences.
        
        Args:
            sentences: List of sentences
            src_lang: Source language
            tgt_lang: Target language
            
        Returns:
            List of sentences with translated text
        """
        # Create segments format for translation
        segments = [{'text': sentence['text']} for sentence in sentences]
        
        # Translate
        translated_segments = translate_segments(segments, src_lang, tgt_lang, debug=True)
        
        # Add translated text to sentences
        for i, sentence in enumerate(sentences):
            sentence['translated_text'] = translated_segments[i]['text']
        
        return sentences
    
    def _clone_sentence_voices(self, sentences: List[Dict], audio_segments: List[str], 
                             output_dir: str, language: str) -> List[str]:
        """
        Voice clone each sentence with its audio reference.
        
        Args:
            sentences: List of sentences with translated text
            audio_segments: List of audio segment files
            output_dir: Directory to save voice-cloned files
            language: Target language
            
        Returns:
            List of paths to voice-cloned audio files
        """
        voice_cloned_files = []
        
        for i, sentence in enumerate(sentences):
            if i < len(audio_segments) and os.path.exists(audio_segments[i]):
                translated_text = sentence['translated_text']
                reference_audio = audio_segments[i]
                
                output_file = os.path.join(output_dir, f"cloned_{i:04d}.wav")
                
                try:
                    # Voice clone this sentence
                    self.voice_cloner.clone_voice(
                        text=translated_text,
                        audio_path=reference_audio,
                        output_path=output_file,
                        language=language
                    )
                    
                    voice_cloned_files.append(output_file)
                    logger.info(f"Voice cloned sentence {i+1}: {translated_text[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Failed to voice clone sentence {i+1}: {e}")
                    # Add empty file or skip
                    voice_cloned_files.append("")
            else:
                logger.warning(f"No audio reference for sentence {i+1}")
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
        # For now, we'll extract gaps between voice segments
        # In a more sophisticated version, you could use music detection
        non_voice_files = []
        
        # Find gaps between segments
        gaps = []
        current_time = 0.0
        
        for segment in segments:
            if segment['start'] > current_time:
                # Found a gap
                gap_duration = segment['start'] - current_time
                if gap_duration > 0.5:  # Only extract gaps longer than 0.5s
                    gaps.append({
                        'start': current_time,
                        'end': segment['start'],
                        'duration': gap_duration
                    })
            current_time = max(current_time, segment['end'])
        
        # Extract gap audio
        for i, gap in enumerate(gaps):
            output_file = os.path.join(output_dir, f"gap_{i:04d}.wav")
            
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
                logger.info(f"Extracted gap {i+1}: {gap['start']:.2f}s - {gap['end']:.2f}s ({gap['duration']:.2f}s)")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract gap {i+1}: {e}")
        
        return non_voice_files
    
    def _reconstruct_audio(self, voice_files: List[str], non_voice_files: List[str],
                         sentences: List[Dict], original_segments: List[Dict], output_path: str) -> str:
        """
        Reconstruct final audio with original timing.
        
        Args:
            voice_files: List of voice-cloned audio files
            non_voice_files: List of non-voice audio files
            sentences: List of sentences with timing info
            original_segments: Original Whisper segments
            output_path: Path for output audio file
            
        Returns:
            Path to reconstructed audio file
        """
        # Create a timeline of all audio elements
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
                    'sentence': sentences[i] if i < len(sentences) else None
                })
        
        # Create concat file
        temp_list_file = os.path.join(os.path.dirname(output_path), "multi_speaker_concat_list.txt")
        
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


def translate_multi_speaker_audio(audio_path: str,
                                src_lang: str = "en",
                                tgt_lang: str = "es",
                                output_path: Optional[str] = None,
                                progress_hook: Optional[Callable[[int, int, str], None]] = None) -> str:
    """
    Convenience function to translate multi-speaker audio file.
    
    Args:
        audio_path: Path to input audio file
        src_lang: Source language code
        tgt_lang: Target language code
        output_path: Path for output audio file
        progress_hook: Optional callback for progress updates
        
    Returns:
        Path to translated audio file
    """
    translator = MultiSpeakerTranslator()
    return translator.translate_audio(
        audio_path=audio_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        output_path=output_path,
        progress_hook=progress_hook
    ) 