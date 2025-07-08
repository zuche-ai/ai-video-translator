"""
Audio Translator Module

This module provides functionality to translate audio files by:
1. Transcribing audio with Whisper
2. Using VAD to identify voice/non-voice segments
3. Translating the transcription
4. Modifying SRT timestamps to exclude non-voice segments
5. Generating voice-cloned audio for voice segments
6. Stitching all segments into a continuous audio file
"""

import os
import tempfile
import logging
import webrtcvad
import numpy as np
import librosa
from typing import List, Dict, Any, Tuple, Optional, Callable
import subprocess

# Import existing modules
from video_translator.core.transcriber import transcribe_video
from video_translator.core.translator import translate_segments
from video_translator.audio.voice_cloner import VoiceCloner

logger = logging.getLogger(__name__)


class AudioTranslator:
    """Main class for audio translation workflow."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the audio translator.
        
        Args:
            temp_dir: Directory for temporary files (defaults to system temp dir)
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.voice_cloner = VoiceCloner()
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (medium)
        
    def translate_audio(self, 
                       audio_path: str,
                       src_lang: str = "en",
                       tgt_lang: str = "es",
                       reference_audio_path: Optional[str] = None,
                       output_path: Optional[str] = None,
                       progress_hook: Optional[Callable[[int, int, str], None]] = None) -> str:
        """
        Main method to translate audio file.
        
        Args:
            audio_path: Path to input audio file
            src_lang: Source language code
            tgt_lang: Target language code
            reference_audio_path: Path to reference audio for voice cloning
            output_path: Path for output audio file
            progress_hook: Optional callback for progress updates
            
        Returns:
            Path to translated audio file
        """
        if progress_hook:
            progress_hook(0, 5, "Starting audio translation...")
        
        # Create temporary directories
        temp_base = os.path.join(self.temp_dir, f"audio_translate_{os.getpid()}")
        voice_segments_dir = os.path.join(temp_base, "voice_segments")
        non_voice_segments_dir = os.path.join(temp_base, "non_voice_segments")
        final_audio_dir = os.path.join(temp_base, "final_audio")
        
        for dir_path in [voice_segments_dir, non_voice_segments_dir, final_audio_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            # Step 1: Transcribe audio with Whisper
            if progress_hook:
                progress_hook(1, 15, "Transcribing audio...")
            
            segments = transcribe_video(audio_path, language=src_lang, debug=True)
            logger.info(f"Transcription completed with {len(segments)} segments")
            
            # Step 2: Use VAD to identify voice/non-voice segments
            if progress_hook:
                progress_hook(2, 25, "Analyzing voice activity...")
            
            voice_segments, non_voice_segments = self._analyze_voice_activity(
                audio_path, segments
            )
            logger.info(f"VAD analysis: {len(voice_segments)} voice segments, {len(non_voice_segments)} non-voice segments")
            
            # Step 3: Translate the transcription
            if progress_hook:
                progress_hook(3, 35, "Translating text...")
            
            translated_segments = translate_segments(segments, src_lang, tgt_lang, debug=True)
            logger.info(f"Translation completed for {len(translated_segments)} segments")
            
            # Step 4: Generate voice-cloned audio for voice segments (using original segments)
            if progress_hook:
                progress_hook(4, 45, "Generating voice-cloned audio...")
            
            voice_audio_files = self._generate_voice_audio(
                translated_segments, reference_audio_path, voice_segments_dir, tgt_lang
            )
            logger.info(f"Voice cloning completed: {len(voice_audio_files)} files generated")
            
            # Step 5: Extract non-voice segments from original audio
            if progress_hook:
                progress_hook(5, 65, "Extracting non-voice segments...")
            
            non_voice_audio_files = self._extract_non_voice_segments(
                audio_path, non_voice_segments, non_voice_segments_dir
            )
            logger.info(f"Non-voice segments extracted: {len(non_voice_audio_files)} files")
            
            # Step 6: Stitch all segments into continuous audio
            if progress_hook:
                progress_hook(6, 85, "Stitching audio segments...")
            
            if not output_path:
                output_path = os.path.join(
                    os.path.dirname(audio_path),
                    f"translated_{os.path.basename(audio_path)}"
                )
            
            final_audio_path = self._stitch_audio_segments(
                voice_audio_files, non_voice_audio_files, 
                voice_segments, non_voice_segments, output_path
            )
            
            if progress_hook:
                progress_hook(7, 100, "Audio translation completed!")
            
            logger.info(f"Audio translation completed: {final_audio_path}")
            return final_audio_path
            
        except Exception as e:
            logger.error(f"Audio translation failed: {e}")
            raise
        finally:
            # Clean up temporary files
            self._cleanup_temp_files(temp_base)
    
    def _analyze_voice_activity(self, audio_path: str, segments: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyze voice activity in audio file using VAD across the entire timeline.
        
        Args:
            audio_path: Path to audio file
            segments: Whisper transcription segments
            
        Returns:
            Tuple of (voice_segments, non_voice_segments)
        """
        # Load audio
        audio, sample_rate = librosa.load(audio_path, sr=16000)  # VAD requires 16kHz
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # VAD frame size (30ms at 16kHz = 480 samples)
        frame_size = int(0.03 * sample_rate)
        
        # Step 1: Run VAD across the entire audio timeline
        vad_regions = self._detect_vad_regions(audio_int16, int(sample_rate), frame_size)
        
        # Debug: Log VAD regions
        logger.info(f"VAD detected {len(vad_regions)} regions:")
        for i, region in enumerate(vad_regions):
            logger.info(f"  Region {i+1}: {region['start']:.2f}s - {region['end']:.2f}s ({'voice' if region['is_voice'] else 'non-voice'})")
        
        # Step 2: Map Whisper segments to VAD regions (keep all as voice segments for translation)
        #TODO: bug here with how it calculates time ranges for voice segments
        #voice_segments, _ = self._map_segments_to_vad(segments, vad_regions)
        
        # Step 3: Create non-voice segments from pure VAD non-voice regions
        non_voice_segments = self._create_non_voice_segments_from_vad(vad_regions)
        
        return segments, non_voice_segments
    
    def _create_non_voice_segments_from_vad(self, vad_regions: List[Dict]) -> List[Dict]:
        """
        Create non-voice segments from VAD regions that are marked as non-voice.
        
        Args:
            vad_regions: List of VAD regions with voice activity status
            
        Returns:
            List of non-voice segments
        """
        non_voice_segments = []
        
        for region in vad_regions:
            if not region['is_voice']:
                # Create a non-voice segment from this VAD region
                non_voice_segments.append({
                    'start': region['start'],
                    'end': region['end'],
                    'text': '',  # No text for non-voice segments
                    'type': 'non_voice'
                })
        
        logger.info(f"Created {len(non_voice_segments)} non-voice segments from VAD regions")
        return non_voice_segments
    
    def _detect_vad_regions(self, audio_int16: np.ndarray, sample_rate: int, frame_size: int) -> List[Dict]:
        """
        Detect voice activity regions across the entire audio timeline.
        
        Args:
            audio_int16: 16-bit PCM audio data
            sample_rate: Audio sample rate
            frame_size: VAD frame size in samples
            
        Returns:
            List of VAD regions with start/end times and voice activity status
        """
        vad_regions = []
        current_region_start = 0
        current_is_voice = None
        
        # Analyze frames
        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[i:i + frame_size]
            if len(frame) == frame_size:
                is_speech = self.vad.is_speech(frame.tobytes(), sample_rate)
                frame_time = i / sample_rate
                
                # If this is the first frame or voice activity changed
                if current_is_voice is None:
                    current_is_voice = is_speech
                elif current_is_voice != is_speech:
                    # Voice activity changed, save the previous region
                    region_end_time = frame_time
                    vad_regions.append({
                        'start': current_region_start,
                        'end': region_end_time,
                        'is_voice': current_is_voice
                    })
                    
                    # Start new region
                    current_region_start = frame_time
                    current_is_voice = is_speech
        
        # Add the final region
        if current_is_voice is not None:
            final_time = len(audio_int16) / sample_rate
            vad_regions.append({
                'start': current_region_start,
                'end': final_time,
                'is_voice': current_is_voice
            })
        
        # Log original VAD regions before merging
        logger.info(f"Original VAD regions before merging:")
        for i, region in enumerate(vad_regions):
            duration = region['end'] - region['start']
            logger.info(f"  Region {i+1}: {region['start']:.2f}s - {region['end']:.2f}s ({duration:.3f}s, {'voice' if region['is_voice'] else 'non-voice'})")
        
        # Merge very short regions (less than 200ms) to avoid fragmentation
        merged_regions = self._merge_short_vad_regions(vad_regions, min_duration=0.1)
        
        # Log merged VAD regions
        logger.info(f"VAD regions after merging:")
        for i, region in enumerate(merged_regions):
            duration = region['end'] - region['start']
            logger.info(f"  Region {i+1}: {region['start']:.2f}s - {region['end']:.2f}s ({duration:.3f}s, {'voice' if region['is_voice'] else 'non-voice'})")
        
        return merged_regions
    
    def _merge_short_vad_regions(self, regions: List[Dict], min_duration: float = 0.2) -> List[Dict]:
        """
        Merge very short VAD regions to reduce fragmentation.
        
        Args:
            regions: List of VAD regions
            min_duration: Minimum duration in seconds
            
        Returns:
            Merged regions
        """
        if not regions:
            return regions
        
        merged = []
        current = regions[0].copy()
        
        for region in regions[1:]:
            duration = region['end'] - region['start']
            
            if duration < min_duration:
                # Extend current region
                current['end'] = region['end']
            else:
                # Save current region and start new one
                merged.append(current)
                current = region.copy()
        
        # Add the last region
        merged.append(current)
        
        return merged
    
    def _map_segments_to_vad(self, segments: List[Dict], vad_regions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Map Whisper segments to VAD regions to determine voice/non-voice classification.
        
        Args:
            segments: Whisper transcription segments
            vad_regions: VAD regions with voice activity status
            
        Returns:
            Tuple of (voice_segments, non_voice_segments)
        """
        voice_segments = []
        non_voice_segments = []
        
        for segment in segments:
            segment_start = segment['start']
            segment_end = segment['end']
            
            # Find overlapping VAD regions
            overlapping_regions = []
            for region in vad_regions:
                # Check for overlap
                if (segment_start < region['end'] and segment_end > region['start']):
                    overlap_start = max(segment_start, region['start'])
                    overlap_end = min(segment_end, region['end'])
                    overlap_duration = overlap_end - overlap_start
                    
                    overlapping_regions.append({
                        'region': region,
                        'overlap_duration': overlap_duration,
                        'overlap_ratio': overlap_duration / (segment_end - segment_start)
                    })
            
            if not overlapping_regions:
                # No VAD regions found, treat as voice segment
                voice_segments.append(segment)
                continue
            
            # Calculate voice activity ratio for this segment
            total_voice_duration = 0.0
            total_duration = segment_end - segment_start
            
            for overlap in overlapping_regions:
                if overlap['region']['is_voice']:
                    total_voice_duration += overlap['overlap_duration']
            
            voice_ratio = total_voice_duration / total_duration
            
            # Classify segment based on voice ratio
            if voice_ratio > 0.5:  # More than 50% voice activity
                voice_segments.append(segment)
            else:
                non_voice_segments.append(segment)
        
        return voice_segments, non_voice_segments
    

    
    def _generate_voice_audio(self, 
                            segments: List[Dict], 
                            reference_audio_path: Optional[str],
                            output_dir: str,
                            language: str) -> List[str]:
        """
        Generate voice-cloned audio for voice segments.
        
        Args:
            segments: Translated segments
            reference_audio_path: Path to reference audio
            output_dir: Directory to save audio files
            language: Target language
            
        Returns:
            List of paths to generated audio files
        """
        if not reference_audio_path:
            raise ValueError("Reference audio path is required for voice cloning")
        
        audio_files = []
        texts = []
        
        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            if text:
                texts.append(text)
        
        # Generate audio files using voice cloner
        audio_files = self.voice_cloner.batch_clone_voice(
            reference_audio_path=reference_audio_path,
            texts=texts,
            output_dir=output_dir,
            language=language
        )
        
        return audio_files
    
    def _extract_non_voice_segments(self, 
                                  audio_path: str,
                                  non_voice_segments: List[Dict],
                                  output_dir: str) -> List[str]:
        """
        Extract non-voice segments from original audio.
        
        Args:
            audio_path: Path to original audio file
            non_voice_segments: Non-voice segments
            output_dir: Directory to save extracted segments
            
        Returns:
            List of paths to extracted audio files
        """
        audio_files = []
        
        for i, segment in enumerate(non_voice_segments):
            start_time = segment['start']
            end_time = segment['end']
            
            output_file = os.path.join(output_dir, f"non_voice_{i:04d}.wav")
            
            # Use ffmpeg to extract segment
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c', 'copy',
                output_file
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                audio_files.append(output_file)
                logger.info(f"Extracted non-voice segment {i}: {start_time:.2f}s - {end_time:.2f}s")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract non-voice segment {i}: {e}")
        
        return audio_files
    
    def _stitch_audio_segments(self,
                             voice_files: List[str],
                             non_voice_files: List[str],
                             voice_segments: List[Dict],
                             non_voice_segments: List[Dict],
                             output_path: str) -> str:
        """
        Stitch voice and non-voice audio segments into continuous audio.
        
        Args:
            voice_files: List of voice audio files
            non_voice_files: List of non-voice audio files
            voice_segments: Voice segments with timing info
            non_voice_segments: Non-voice segments with timing info
            output_path: Path for output audio file
            
        Returns:
            Path to stitched audio file
        """
        # Create a temporary file list for ffmpeg
        temp_list_file = os.path.join(os.path.dirname(output_path), "concat_list.txt")
        
        # TODO: THERE'S A BUG HERE.
        # Create a unified timeline of all segments with proper ordering
        timeline_segments = self._create_timeline_segments(
            voice_files, non_voice_files, voice_segments, non_voice_segments
        )
        
        logger.info(f"Stitching {len(timeline_segments)} segments in chronological order:")
        for i, segment in enumerate(timeline_segments):
            logger.info(f"  {i+1}. {segment['type']}: {segment['original_start']:.2f}s - {segment['original_end']:.2f}s -> {segment['file']}")
        
        # Create concat file
        with open(temp_list_file, 'w') as f:
            for segment in timeline_segments:
                f.write(f"file '{segment['file']}'\n")
        
        # Use ffmpeg to concatenate all files
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list_file,
            '-c', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Audio stitching completed: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stitch audio: {e}")
            raise RuntimeError(f"Audio stitching failed: {e}")
        finally:
            # Clean up temp list file
            if os.path.exists(temp_list_file):
                os.remove(temp_list_file)
        
        return output_path
    
    def _create_timeline_segments(self,
                                voice_files: List[str],
                                non_voice_files: List[str],
                                voice_segments: List[Dict],
                                non_voice_segments: List[Dict]) -> List[Dict]:
        """
        Create a timeline-based insertion map for segments without modifying original timings.
        
        Args:
            voice_files: List of voice audio files
            non_voice_files: List of non-voice audio files
            voice_segments: Voice segments with timing info
            non_voice_segments: Non-voice segments with timing info
            
        Returns:
            List of segments with insertion timeline
        """
        # Create insertion timeline
        timeline = self._create_insertion_timeline(voice_segments, non_voice_segments)
        
        # Map files to timeline entries
        timeline_segments = []
        
        # Add voice segments with their original timings preserved
        for i, segment in enumerate(voice_segments):
            if i < len(voice_files):
                timeline_segments.append({
                    'type': 'voice',
                    'file': voice_files[i],
                    'original_start': segment.get('start', 0),
                    'original_end': segment.get('end', 0),
                    'insertion_time': segment.get('start', 0),  # Voice segments keep original timing
                    'duration': segment.get('end', 0) - segment.get('start', 0)
                })
        
        # Add non-voice segments with their insertion times from timeline
        for i, segment in enumerate(non_voice_segments):
            if i < len(non_voice_files):
                insertion_time = timeline.get(f"non_voice_{i}", segment.get('start', 0))
                timeline_segments.append({
                    'type': 'non_voice',
                    'file': non_voice_files[i],
                    'original_start': segment.get('start', 0),
                    'original_end': segment.get('end', 0),
                    'insertion_time': insertion_time,
                    'duration': segment.get('end', 0) - segment.get('start', 0)
                })
        
        # Sort by insertion time for proper ordering
        timeline_segments.sort(key=lambda x: x['insertion_time'])
        
        logger.info(f"Created timeline with {len(timeline_segments)} segments:")
        for i, segment in enumerate(timeline_segments):
            logger.info(f"  {i+1}. {segment['type']}: insert at {segment['insertion_time']:.2f}s (original: {segment['original_start']:.2f}s - {segment['original_end']:.2f}s)")
        
        return timeline_segments
    
    def _create_insertion_timeline(self, voice_segments: List[Dict], non_voice_segments: List[Dict]) -> Dict[str, float]:
        """
        Create an insertion timeline that determines when non-voice segments should be inserted.
        
        Args:
            voice_segments: Voice segments with timing info
            non_voice_segments: Non-voice segments with timing info
            
        Returns:
            Dictionary mapping non-voice segment indices to insertion times
        """
        timeline = {}
        
        # Create a timeline of all voice segments
        voice_timeline = []
        for segment in voice_segments:
            voice_timeline.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0)
            })
        
        # Sort voice timeline by start time
        voice_timeline.sort(key=lambda x: x['start'])
        
        # Find gaps between voice segments where non-voice can be inserted
        gaps = []
        current_time = 0.0
        
        for voice_seg in voice_timeline:
            if voice_seg['start'] > current_time:
                # Found a gap
                gaps.append({
                    'start': current_time,
                    'end': voice_seg['start'],
                    'duration': voice_seg['start'] - current_time
                })
            current_time = max(current_time, voice_seg['end'])
        
        # Add final gap if there's space after the last voice segment
        if voice_timeline:
            last_voice_end = max(seg['end'] for seg in voice_timeline)
            gaps.append({
                'start': last_voice_end,
                'end': float('inf'),  # Will be adjusted based on non-voice segments
                'duration': float('inf')
            })
        
        # Sort non-voice segments by their original start time
        sorted_non_voice = sorted(enumerate(non_voice_segments), key=lambda x: x[1].get('start', 0))
        
        # Assign non-voice segments to gaps
        gap_index = 0
        for non_voice_idx, non_voice_seg in sorted_non_voice:
            non_voice_start = non_voice_seg.get('start', 0)
            non_voice_end = non_voice_seg.get('end', 0)
            non_voice_duration = non_voice_end - non_voice_start
            
            # Find the best gap for this non-voice segment
            best_gap = None
            best_gap_idx = None
            
            for i, gap in enumerate(gaps[gap_index:], gap_index):
                if gap['duration'] >= non_voice_duration:
                    # This gap can fit the non-voice segment
                    best_gap = gap
                    best_gap_idx = i
                    break
            
            if best_gap:
                # Insert non-voice segment at the start of this gap
                insertion_time = best_gap['start']
                timeline[f"non_voice_{non_voice_idx}"] = insertion_time
                
                # Update the gap
                remaining_duration = best_gap['duration'] - non_voice_duration
                if remaining_duration > 0:
                    # Split the gap
                    best_gap['start'] = insertion_time + non_voice_duration
                    best_gap['duration'] = remaining_duration
                else:
                    # Gap is fully used, remove it
                    if best_gap_idx is not None:
                        gaps.pop(best_gap_idx)
                
                logger.info(f"Assigned non-voice segment {non_voice_idx} to gap {best_gap_idx}: insert at {insertion_time:.2f}s")
            else:
                # No suitable gap found, insert at original time
                timeline[f"non_voice_{non_voice_idx}"] = non_voice_start
                logger.warning(f"No suitable gap found for non-voice segment {non_voice_idx}, using original time {non_voice_start:.2f}s")
        
        return timeline
    
    def _resolve_segment_overlaps(self, segments: List[Dict]) -> List[Dict]:
        """
        Resolve overlaps between voice and non-voice segments by adjusting voice segment timings.
        
        Args:
            segments: List of segments with timing info
            
        Returns:
            List of segments with resolved overlaps
        """
        # Separate voice and non-voice segments
        voice_segments = [s for s in segments if s['type'] == 'voice']
        non_voice_segments = [s for s in segments if s['type'] == 'non_voice']
        
        # Create a list of non-voice time ranges
        non_voice_ranges = [(s['original_start'], s['original_end']) for s in non_voice_segments]
        
        # Adjust voice segments to avoid overlaps
        adjusted_voice_segments = []
        for voice_seg in voice_segments:
            adjusted_seg = voice_seg.copy()
            start_time = voice_seg['original_start']
            end_time = voice_seg['original_end']
            
            # Check for overlaps with non-voice segments
            for non_voice_start, non_voice_end in non_voice_ranges:
                # If there's an overlap
                if start_time < non_voice_end and end_time > non_voice_start:
                    # Adjust start time if voice segment starts before non-voice
                    if start_time < non_voice_start:
                        # Voice segment starts before non-voice, adjust end time
                        adjusted_seg['original_end'] = non_voice_start
                        logger.info(f"Adjusted voice segment end: {end_time:.2f}s -> {non_voice_start:.2f}s")
                    elif end_time > non_voice_end:
                        # Voice segment ends after non-voice, adjust start time
                        adjusted_seg['original_start'] = non_voice_end
                        logger.info(f"Adjusted voice segment start: {start_time:.2f}s -> {non_voice_end:.2f}s")
                    else:
                        # Voice segment is completely contained within non-voice segment
                        # This shouldn't happen with proper VAD, but handle it
                        logger.warning(f"Voice segment completely within non-voice segment: {start_time:.2f}s - {end_time:.2f}s")
                        continue
            
            # Only add if the segment still has positive duration
            if adjusted_seg['original_end'] > adjusted_seg['original_start']:
                adjusted_voice_segments.append(adjusted_seg)
            else:
                logger.warning(f"Removing voice segment with zero/negative duration: {adjusted_seg['original_start']:.2f}s - {adjusted_seg['original_end']:.2f}s")
        
        # Combine adjusted voice segments with non-voice segments
        result = adjusted_voice_segments + non_voice_segments
        
        logger.info(f"Resolved overlaps: {len(voice_segments)} voice segments -> {len(adjusted_voice_segments)} after adjustment")
        
        return result
    
    def _cleanup_temp_files(self, temp_base: str):
        """Clean up temporary files and directories."""
        try:
            import shutil
            if os.path.exists(temp_base):
                shutil.rmtree(temp_base)
                logger.info(f"Cleaned up temporary directory: {temp_base}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")


def translate_audio_file(audio_path: str,
                        src_lang: str = "en",
                        tgt_lang: str = "es",
                        reference_audio_path: Optional[str] = None,
                        output_path: Optional[str] = None,
                        progress_hook: Optional[Callable[[int, int, str], None]] = None) -> str:
    """
    Convenience function to translate an audio file.
    
    Args:
        audio_path: Path to input audio file
        src_lang: Source language code
        tgt_lang: Target language code
        reference_audio_path: Path to reference audio for voice cloning
        output_path: Path for output audio file
        progress_hook: Optional callback for progress updates
        
    Returns:
        Path to translated audio file
    """
    translator = AudioTranslator()
    return translator.translate_audio(
        audio_path=audio_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        reference_audio_path=reference_audio_path,
        output_path=output_path,
        progress_hook=progress_hook
    ) 