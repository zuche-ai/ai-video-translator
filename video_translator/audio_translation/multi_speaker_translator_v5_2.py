#!/usr/bin/env python3
"""
Multi-Speaker Audio Translator V5.2
Balanced sentence merging with pause detection and non-voice preservation
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
import whisper
import torch
from typing import List, Dict, Tuple, Optional
import numpy as np
import librosa
import re

# Add the parent directory to the path to import from video_translator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.transcriber import transcribe_video
from core.translator import translate_segments
from audio.voice_cloner import VoiceCloner

class MultiSpeakerTranslatV5_2:
    def __init__(self, 
                 source_language: str = "en",
                 target_language: str = "es",
                 model_name: str = "base",
                 voice_clone_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.source_language = source_language
        self.target_language = target_language
        self.model_name = model_name
        self.device = device
        
        # Initialize components
        self.voice_cloner = VoiceCloner(model_name=voice_clone_model)
        
    def translate_audio(self, audio_path: str, output_path: str):
        """Main translation pipeline with balanced merging and pause detection"""
        print(f"🎯 V5.2: Balanced Fluid Voice Translation with Pause Detection")
        print(f"📁 Input: {audio_path}")
        print(f"📁 Output: {output_path}")
        
        # Get audio duration
        duration = self._get_audio_duration(audio_path)
        print(f"⏱️ Audio duration: {duration:.2f}s")
        
        # Get detailed Whisper segments
        print("📝 Getting detailed Whisper segments...")
        segments = transcribe_video(audio_path, self.source_language, word_timestamps=True)
        
        if not segments:
            print("⚠️ No segments found, treating entire audio as voice")
            return self._handle_single_voice_segment(audio_path, output_path, duration)
        
        print(f"📊 Found {len(segments)} initial segments")
        
        # Balanced merging that preserves non-voice segments
        fluid_segments = self._merge_into_balanced_sentences(segments, audio_path)
        print(f"🔄 Merged into {len(fluid_segments)} balanced segments")
        
        # Enhanced classification with non-voice preservation
        voice_segments, non_voice_segments = self._classify_segments_with_preservation(fluid_segments, segments, audio_path)
        
        print(f"🎤 Voice segments: {len(voice_segments)}")
        print(f"🎵 Non-voice segments: {len(non_voice_segments)}")
        
        # Process voice segments with pause detection
        processed_voice_segments = self._process_voice_segments_with_pause_detection(voice_segments, segments, audio_path)
        
        # Reconstruct audio with pause optimization
        self._reconstruct_audio_with_pause_optimization(processed_voice_segments, non_voice_segments, output_path)
        
        print(f"✅ Translation complete! Output: {output_path}")
        
    def _merge_into_balanced_sentences(self, segments: List[Dict], audio_path: str) -> List[Dict]:
        """Balanced merging that preserves non-voice segments"""
        print("🔄 Balanced merging with non-voice preservation...")
        
        if not segments:
            return []
        
        # First pass: Conservative merging
        merged_segments = self._conservative_merge_pass(segments)
        print(f"📊 First pass: {len(merged_segments)} segments")
        
        # Second pass: Smart merging with non-voice detection
        merged_segments = self._smart_merge_pass(merged_segments, audio_path)
        print(f"📊 Second pass: {len(merged_segments)} segments")
        
        # Extract audio for each merged segment
        for segment in merged_segments:
            segment['audio_path'] = self._extract_audio_segment(
                audio_path, segment['start'], segment['end']
            )
        
        return merged_segments
    
    def _conservative_merge_pass(self, segments: List[Dict]) -> List[Dict]:
        """Conservative merging that only merges obvious continuations"""
        merged_segments = []
        current_segment = None
        
        for segment in segments:
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            if current_segment is None:
                current_segment = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': text,
                    'speaker': segment.get('speaker', 'speaker_0'),
                    'segments': [segment]
                }
                continue
            
            # Conservative merge conditions
            should_merge = (
                segment.get('speaker') == current_segment.get('speaker') and
                segment['start'] - current_segment['end'] <= 1.0 and  # Very small gap
                (
                    not self._is_complete_sentence(current_segment['text']) or
                    self._has_obvious_continuation(current_segment['text'], text)
                )
            )
            
            if should_merge:
                current_segment['end'] = segment['end']
                current_segment['text'] += ' ' + text
                current_segment['segments'].append(segment)
            else:
                merged_segments.append(current_segment)
                current_segment = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': text,
                    'speaker': segment.get('speaker', 'speaker_0'),
                    'segments': [segment]
                }
        
        if current_segment:
            merged_segments.append(current_segment)
        
        return merged_segments
    
    def _smart_merge_pass(self, segments: List[Dict], audio_path: str) -> List[Dict]:
        """Smart merging that detects and preserves non-voice segments"""
        if len(segments) <= 1:
            return segments
        
        merged_segments = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # Check if current segment might be non-voice
            current_duration = current['end'] - current['start']
            if current_duration < 2.0:  # Short segment
                # Analyze if it's likely non-voice
                is_likely_non_voice = self._is_likely_non_voice_segment(current, audio_path)
                if is_likely_non_voice:
                    # Don't merge, preserve as separate segment
                    merged_segments.append(current)
                    i += 1
                    continue
            
            # Look ahead for safe merging
            j = i + 1
            while j < len(segments):
                next_seg = segments[j]
                
                # Safe merge conditions
                can_merge = (
                    next_seg.get('speaker') == current.get('speaker') and
                    next_seg['start'] - current['end'] <= 1.5 and
                    not self._is_likely_non_voice_segment(next_seg, audio_path) and
                    (
                        len(current['text'].split()) < 6 or  # Short segment
                        not self._is_complete_sentence(current['text']) or
                        self._has_obvious_continuation(current['text'], next_seg['text'])
                    )
                )
                
                if can_merge:
                    current['end'] = next_seg['end']
                    current['text'] += ' ' + next_seg['text']
                    current['segments'].extend(next_seg['segments'])
                    j += 1
                else:
                    break
            
            merged_segments.append(current)
            i = j
        
        return merged_segments
    
    def _is_likely_non_voice_segment(self, segment: Dict, audio_path: str) -> bool:
        """Quick check if segment is likely non-voice"""
        duration = segment['end'] - segment['start']
        text = segment.get('text', '').strip()
        
        # Very short segments with little text are likely non-voice
        if duration < 1.5 and len(text.split()) <= 2:
            return True
        
        # Segments with only punctuation or filler words
        if text.lower() in ['[music]', '[silence]', '[applause]', '...', '--']:
            return True
        
        return False
    
    def _has_obvious_continuation(self, current_text: str, next_text: str) -> bool:
        """Check for obvious speech continuations"""
        current_lower = current_text.lower().strip()
        next_lower = next_text.lower().strip()
        
        # Obvious continuation indicators
        obvious_indicators = [
            'and', 'but', 'or', 'so', 'because', 'however', 'therefore',
            'meanwhile', 'furthermore', 'moreover', 'nevertheless'
        ]
        
        # Check if current ends with obvious continuation
        for indicator in obvious_indicators:
            if current_lower.endswith(f' {indicator}'):
                return True
        
        # Check if next starts with obvious continuation
        for indicator in obvious_indicators:
            if next_lower.startswith(f'{indicator} '):
                return True
        
        # Check for trailing punctuation that suggests continuation
        if current_lower.endswith((':', ';', ',')):
            return True
        
        return False
    
    def _is_complete_sentence(self, text: str) -> bool:
        """Check if text forms a complete sentence"""
        text = text.strip()
        if not text:
            return False
        
        # Must have at least 3 words
        words = text.split()
        if len(words) < 3:
            return False
        
        # Must end with sentence-ending punctuation
        if not text.rstrip().endswith(('.', '!', '?')):
            return False
        
        return True
    
    def _classify_segments_with_preservation(self, segments: List[Dict], original_segments: List[Dict], audio_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Enhanced classification that preserves non-voice segments"""
        print("🎯 Enhanced classification with non-voice preservation...")
        
        voice_segments = []
        non_voice_segments = []
        
        # Find intro segments (silence + music before first voice)
        intro_segments = self._detect_intro_segments(segments, audio_path)
        non_voice_segments.extend(intro_segments)
        
        # Find gaps between segments that should be preserved as non-voice
        gap_segments = self._detect_gap_segments(segments, audio_path)
        non_voice_segments.extend(gap_segments)
        
        # Classify remaining segments
        for segment in segments:
            if segment in intro_segments or segment in gap_segments:
                continue
                
            # Enhanced voice detection
            is_voice = self._is_voice_segment_enhanced(segment, audio_path)
            
            if is_voice:
                voice_segments.append(segment)
            else:
                non_voice_segments.append(segment)
        
        return voice_segments, non_voice_segments
    
    def _detect_intro_segments(self, segments: List[Dict], audio_path: str) -> List[Dict]:
        """Detect intro silence and music segments"""
        if not segments:
            return []
        
        first_voice_start = segments[0]['start']
        intro_segments = []
        
        # If there's significant time before first voice, extract as intro
        if first_voice_start > 1.0:
            intro_segments.append({
                'start': 0.0,
                'end': first_voice_start,
                'text': '[intro]',
                'speaker': 'intro',
                'type': 'non_voice',
                'audio_path': self._extract_audio_segment(audio_path, 0.0, first_voice_start)
            })
        
        return intro_segments
    
    def _detect_gap_segments(self, segments: List[Dict], audio_path: str) -> List[Dict]:
        """Detect gaps between segments that should be preserved"""
        gap_segments = []
        
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            gap_start = current['end']
            gap_end = next_seg['start']
            gap_duration = gap_end - gap_start
            
            # Preserve gaps longer than 0.5 seconds
            if gap_duration > 0.5:
                gap_segments.append({
                    'start': gap_start,
                    'end': gap_end,
                    'text': '[gap]',
                    'speaker': 'gap',
                    'type': 'non_voice',
                    'audio_path': self._extract_audio_segment(audio_path, gap_start, gap_end)
                })
        
        return gap_segments
    
    def _is_voice_segment_enhanced(self, segment: Dict, audio_path: str) -> bool:
        """Enhanced voice detection with spectral analysis"""
        try:
            # Load audio segment
            audio, sr = librosa.load(segment['audio_path'], sr=None)
            
            if len(audio) == 0:
                return False
            
            # Enhanced spectral analysis for voice vs music detection
            voice_confidence = self._analyze_voice_characteristics(audio, int(sr))
            
            # Balanced threshold for V5.2
            return voice_confidence > 0.3
            
        except Exception as e:
            print(f"⚠️ Error analyzing segment: {e}")
            return True  # Default to voice if analysis fails
    
    def _analyze_voice_characteristics(self, audio: np.ndarray, sr: int) -> float:
        """Analyze audio characteristics to determine if it's voice"""
        if len(audio) == 0:
            return 0.0
        
        # Calculate spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # Voice characteristics
        avg_centroid = np.mean(spectral_centroids)
        avg_rolloff = np.mean(spectral_rolloff)
        low_freq_energy = np.mean(np.abs(audio))
        
        # Calculate voice confidence score
        voice_score = 0.0
        
        # Lower spectral centroid = more likely voice
        if avg_centroid < 2000:
            voice_score += 0.3
        elif avg_centroid < 3000:
            voice_score += 0.2
        
        # Lower spectral rolloff = more likely voice
        if avg_rolloff < 4000:
            voice_score += 0.3
        elif avg_rolloff < 6000:
            voice_score += 0.2
        
        # Higher low-frequency energy = more likely voice
        if low_freq_energy > 0.01:
            voice_score += 0.2
        
        # Duration factor
        duration = len(audio) / sr
        if duration > 0.5:
            voice_score += 0.2
        
        return min(voice_score, 1.0)
    
    def _process_voice_segments_with_pause_detection(self, voice_segments: List[Dict], original_segments: List[Dict], audio_path: str) -> List[Dict]:
        """Process voice segments with pause detection and optimization"""
        print("🎤 Processing voice segments with pause detection...")
        
        processed_segments = []
        
        for i, segment in enumerate(voice_segments):
            print(f"🎤 Processing voice segment {i+1}/{len(voice_segments)}")
            
            # Clean and prepare text
            text = segment['text'].strip()
            if not text:
                continue
            
            # Translate text
            translated_text = translate_segments([segment], self.source_language, self.target_language)[0]['text']
            
            # Get precise reference audio for voice cloning
            reference_audio_path = self._get_precise_reference_audio(segment, original_segments, audio_path)
            
            # Voice clone with precise reference audio
            cloned_audio_path = f"temp_cloned_voice_{i}.wav"
            if reference_audio_path:
                print(f"🎤 Using precise reference audio: {os.path.basename(reference_audio_path)}")
                self.voice_cloner.clone_voice(
                    translated_text, reference_audio_path, cloned_audio_path, language=self.target_language
                )
            else:
                print(f"🎤 Using segment audio as reference")
                self.voice_cloner.clone_voice(
                    translated_text, segment['audio_path'], cloned_audio_path, language=self.target_language
                )
            
            # Detect and optimize pauses in cloned audio
            optimized_audio_path = self._optimize_pauses(cloned_audio_path, segment)
            
            # Update segment with processed data
            processed_segment = segment.copy()
            processed_segment['translated_text'] = translated_text
            processed_segment['cloned_audio_path'] = optimized_audio_path
            processed_segments.append(processed_segment)
        
        return processed_segments
    
    def _optimize_pauses(self, cloned_audio_path: str, original_segment: Dict) -> str:
        """Optimize pauses in cloned audio to match original timing"""
        optimized_path = cloned_audio_path.replace('.wav', '_optimized.wav')
        
        # Get original segment duration
        original_duration = original_segment['end'] - original_segment['start']
        
        # Get cloned audio duration
        cloned_duration = self._get_audio_duration(cloned_audio_path)
        
        # If cloned audio is significantly longer, speed it up slightly
        if cloned_duration > original_duration * 1.2:  # 20% tolerance
            speed_factor = original_duration / cloned_duration
            speed_factor = max(0.8, min(1.2, speed_factor))  # Limit speed change
            
            cmd = [
                'ffmpeg', '-y', '-i', cloned_audio_path,
                '-filter:a', f'atempo={speed_factor}',
                '-c:a', 'pcm_s16le',
                optimized_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Clean up original
            os.remove(cloned_audio_path)
            return optimized_path
        
        return cloned_audio_path
    
    def _get_precise_reference_audio(self, segment: Dict, original_segments: List[Dict], audio_path: str) -> Optional[str]:
        """Get precise reference audio by mapping to original segments"""
        speaker = segment.get('speaker', 'speaker_0')
        
        # Find the original segments that make up this merged segment
        segment_start = segment['start']
        segment_end = segment['end']
        
        # Find original segments that overlap with this merged segment
        matching_original_segments = []
        for orig_seg in original_segments:
            # Check if original segment overlaps with merged segment
            if (orig_seg['start'] < segment_end and orig_seg['end'] > segment_start and
                orig_seg.get('speaker') == speaker):
                matching_original_segments.append(orig_seg)
        
        if matching_original_segments:
            # Use the longest original segment as reference
            longest_original = max(matching_original_segments, key=lambda s: s['end'] - s['start'])
            
            # Extract the reference audio from the original segment
            ref_start = max(longest_original['start'], segment_start)
            ref_end = min(longest_original['end'], segment_end)
            
            if ref_end > ref_start:
                reference_audio_path = self._extract_audio_segment(audio_path, ref_start, ref_end)
                return reference_audio_path
        
        # Fallback: find other segments from the same speaker
        same_speaker_segments = [s for s in original_segments if s.get('speaker') == speaker]
        
        if len(same_speaker_segments) > 1:
            # Use the longest segment from the same speaker as reference
            longest_segment = max(same_speaker_segments, key=lambda s: s['end'] - s['start'])
            return longest_segment['audio_path']
        
        return None
    
    def _extract_audio_segment(self, audio_path: str, start_time: float, end_time: float) -> str:
        """Extract audio segment using ffmpeg"""
        output_path = f"temp_segment_{start_time}_{end_time}.wav"
        
        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', str(start_time), '-to', str(end_time),
            '-c', 'copy', output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    
    def _reconstruct_audio_with_pause_optimization(self, voice_segments: List[Dict], non_voice_segments: List[Dict], output_path: str):
        """Reconstruct audio with pause optimization"""
        print("🔧 Reconstructing audio with pause optimization...")
        
        # Get original audio format
        if voice_segments:
            probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', voice_segments[0]['audio_path']]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            original_format = json.loads(result.stdout)['format']
        
        # Create file list for concatenation
        file_list_path = "temp_concat_list.txt"
        
        with open(file_list_path, 'w') as f:
            # Add all segments in chronological order
            all_segments = voice_segments + non_voice_segments
            all_segments.sort(key=lambda x: x['start'])
            
            for segment in all_segments:
                if 'cloned_audio_path' in segment:
                    # Voice segment - convert to original format
                    converted_path = f"temp_converted_{segment['start']}_{segment['end']}.wav"
                    self._convert_to_original_format(segment['cloned_audio_path'], converted_path, original_format)
                    f.write(f"file '{converted_path}'\n")
                else:
                    # Non-voice segment - already in original format
                    f.write(f"file '{segment['audio_path']}'\n")
        
        # Concatenate all segments
        concat_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', file_list_path, '-c', 'copy', output_path
        ]
        
        subprocess.run(concat_cmd, capture_output=True, check=True)
        
        # Cleanup
        os.remove(file_list_path)
        for segment in voice_segments:
            if 'cloned_audio_path' in segment:
                os.remove(segment['cloned_audio_path'])
                converted_path = f"temp_converted_{segment['start']}_{segment['end']}.wav"
                if os.path.exists(converted_path):
                    os.remove(converted_path)
        
        for segment in non_voice_segments:
            if os.path.exists(segment['audio_path']):
                os.remove(segment['audio_path'])
    
    def _convert_to_original_format(self, input_path: str, output_path: str, original_format: Dict):
        """Convert voice cloned audio to match original format"""
        # Get target format from original audio
        target_sample_rate = original_format.get('sample_rate', '16000')
        target_channels = original_format.get('channels', '1')
        
        # Convert to match original format
        convert_cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ar', target_sample_rate,
            '-ac', target_channels,
            '-c:a', 'pcm_s16le',
            output_path
        ]
        
        subprocess.run(convert_cmd, capture_output=True, check=True)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe"""
        cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    
    def _handle_single_voice_segment(self, audio_path: str, output_path: str, duration: float):
        """Handle case where entire audio is treated as voice"""
        print("🎤 Treating entire audio as voice segment...")
        
        # Extract audio
        segment_audio_path = self._extract_audio_segment(audio_path, 0.0, duration)
        
        # Get transcription
        segments = transcribe_video(audio_path, self.source_language)
        text = ' '.join([s.get('text', '') for s in segments]) if segments else ''
        
        if not text:
            print("⚠️ No text found, copying original audio")
            shutil.copy2(audio_path, output_path)
            return
        
        # Translate
        translated_text = translate_segments([{'text': text}], self.source_language, self.target_language)[0]['text']
        
        # Voice clone
        cloned_audio_path = "temp_cloned_voice_single.wav"
        self.voice_cloner.clone_voice(translated_text, segment_audio_path, cloned_audio_path, language=self.target_language)
        
        # Convert to original format and save
        self._convert_to_original_format(cloned_audio_path, output_path, {'sample_rate': '16000', 'channels': '1'})
        
        # Cleanup
        os.remove(segment_audio_path)
        os.remove(cloned_audio_path)

def main():
    if len(sys.argv) != 3:
        print("Usage: python multi_speaker_translator_v5_2.py <input_audio> <output_audio>")
        sys.exit(1)
    
    input_audio = sys.argv[1]
    output_audio = sys.argv[2]
    
    translator = MultiSpeakerTranslatV5_2()
    translator.translate_audio(input_audio, output_audio)

if __name__ == "__main__":
    main() 