#!/usr/bin/env python3
"""
Multi-Speaker Audio Translator V5
Enhanced sentence merging for fluid voice segments
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

class MultiSpeakerTranslatV5:
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
        """Main translation pipeline with fluid sentence merging"""
        print(f"🎯 V5: Enhanced Fluid Voice Translation")
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
        
        # Merge segments into fluid sentences
        fluid_segments = self._merge_into_fluid_sentences(segments, audio_path)
        print(f"🔄 Merged into {len(fluid_segments)} fluid segments")
        
        # Classify segments as voice or non-voice
        voice_segments, non_voice_segments = self._classify_segments_enhanced(fluid_segments, audio_path)
        
        print(f"🎤 Voice segments: {len(voice_segments)}")
        print(f"🎵 Non-voice segments: {len(non_voice_segments)}")
        
        # Process voice segments with translation and cloning
        processed_voice_segments = self._process_voice_segments_fluid(voice_segments, audio_path)
        
        # Reconstruct audio with unified format
        self._reconstruct_audio_unified(processed_voice_segments, non_voice_segments, output_path)
        
        print(f"✅ Translation complete! Output: {output_path}")
        
    def _merge_into_fluid_sentences(self, segments: List[Dict], audio_path: str) -> List[Dict]:
        """Merge Whisper segments into complete, fluid sentences"""
        print("🔄 Merging segments into fluid sentences...")
        
        if not segments:
            return []
            
        merged_segments = []
        current_segment = None
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            # If this is the first segment, start a new one
            if current_segment is None:
                current_segment = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': text,
                    'speaker': segment.get('speaker', 'speaker_0'),
                    'segments': [segment]
                }
                continue
            
            # Check if we should merge with current segment
            should_merge = self._should_merge_segments(current_segment, segment)
            
            if should_merge:
                # Merge into current segment
                current_segment['end'] = segment['end']
                current_segment['text'] += ' ' + text
                current_segment['segments'].append(segment)
            else:
                # Finalize current segment and start new one
                merged_segments.append(current_segment)
                current_segment = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': text,
                    'speaker': segment.get('speaker', 'speaker_0'),
                    'segments': [segment]
                }
        
        # Add the last segment
        if current_segment:
            merged_segments.append(current_segment)
        
        # Extract audio for each merged segment
        for segment in merged_segments:
            segment['audio_path'] = self._extract_audio_segment(
                audio_path, segment['start'], segment['end']
            )
        
        return merged_segments
    
    def _should_merge_segments(self, current: Dict, next_segment: Dict) -> bool:
        """Determine if segments should be merged for fluid speech"""
        current_text = current['text'].strip()
        next_text = next_segment.get('text', '').strip()
        
        # Don't merge if there's a significant time gap (>2 seconds)
        time_gap = next_segment['start'] - current['end']
        if time_gap > 2.0:
            return False
        
        # Don't merge if speakers are different
        if current.get('speaker') != next_segment.get('speaker'):
            return False
        
        # Merge if current text doesn't end with sentence-ending punctuation
        if not self._ends_with_sentence_punctuation(current_text):
            return True
        
        # Don't merge if current text ends with sentence punctuation and next starts with capital
        if (self._ends_with_sentence_punctuation(current_text) and 
            next_text and next_text[0].isupper()):
            return False
        
        # Merge if the combined text would be too short (<3 words)
        combined_text = current_text + ' ' + next_text
        if len(combined_text.split()) < 3:
            return True
        
        # Merge if there's a natural speech flow indicator
        if self._has_natural_flow(current_text, next_text):
            return True
        
        return False
    
    def _ends_with_sentence_punctuation(self, text: str) -> bool:
        """Check if text ends with sentence-ending punctuation"""
        return text.rstrip().endswith(('.', '!', '?', ':', ';'))
    
    def _has_natural_flow(self, current_text: str, next_text: str) -> bool:
        """Check if there's natural speech flow between segments"""
        # Check for incomplete phrases
        incomplete_indicators = [
            'and', 'but', 'or', 'so', 'because', 'however', 'therefore',
            'meanwhile', 'furthermore', 'moreover', 'nevertheless'
        ]
        
        current_lower = current_text.lower().strip()
        next_lower = next_text.lower().strip()
        
        # If current ends with a conjunction, merge
        for indicator in incomplete_indicators:
            if current_lower.endswith(f' {indicator}'):
                return True
        
        # If next starts with a conjunction, merge
        for indicator in incomplete_indicators:
            if next_lower.startswith(f'{indicator} '):
                return True
        
        # Check for incomplete sentences (no subject-verb-object pattern)
        if not self._is_complete_sentence(current_text):
            return True
        
        return False
    
    def _is_complete_sentence(self, text: str) -> bool:
        """Basic check if text forms a complete sentence"""
        # Simple heuristic: check for basic sentence structure
        words = text.split()
        if len(words) < 3:
            return False
        
        # Check if it has both a subject (noun/pronoun) and verb
        # This is a simplified check - in practice, you might want more sophisticated NLP
        return True  # For now, assume it's complete if it's long enough
    
    def _classify_segments_enhanced(self, segments: List[Dict], audio_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Enhanced classification with better voice/music detection"""
        print("🎯 Enhanced segment classification...")
        
        voice_segments = []
        non_voice_segments = []
        
        # Find intro segments (silence + music before first voice)
        intro_segments = self._detect_intro_segments(segments, audio_path)
        non_voice_segments.extend(intro_segments)
        
        # Classify remaining segments
        for segment in segments:
            if segment in intro_segments:
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
    
    def _is_voice_segment_enhanced(self, segment: Dict, audio_path: str) -> bool:
        """Enhanced voice detection with spectral analysis"""
        try:
            # Load audio segment
            audio, sr = librosa.load(segment['audio_path'], sr=None)
            
            if len(audio) == 0:
                return False
            
            # Enhanced spectral analysis for voice vs music detection
            voice_confidence = self._analyze_voice_characteristics(audio, int(sr))
            
            # More sensitive threshold for V5
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
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Voice characteristics
        # Voice typically has lower spectral centroid and rolloff than music
        avg_centroid = np.mean(spectral_centroids)
        avg_rolloff = np.mean(spectral_rolloff)
        
        # Voice has more energy in lower frequencies
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
        
        # Duration factor (very short segments are less likely to be voice)
        duration = len(audio) / sr
        if duration > 0.5:
            voice_score += 0.2
        
        return min(voice_score, 1.0)
    
    def _process_voice_segments_fluid(self, voice_segments: List[Dict], audio_path: str) -> List[Dict]:
        """Process voice segments with fluid sentence handling"""
        print("🎤 Processing fluid voice segments...")
        
        processed_segments = []
        
        for i, segment in enumerate(voice_segments):
            print(f"🎤 Processing voice segment {i+1}/{len(voice_segments)}")
            
            # Clean and prepare text
            text = segment['text'].strip()
            if not text:
                continue
            
            # Translate text
            translated_text = translate_segments([segment], self.source_language, self.target_language)[0]['text']
            
            # Get reference audio for voice cloning
            reference_audio_path = self._get_reference_audio(segment, voice_segments, audio_path)
            
            # Voice clone with reference audio
            cloned_audio_path = f"temp_cloned_voice_{i}.wav"
            if reference_audio_path:
                self.voice_cloner.clone_voice(
                    translated_text, reference_audio_path, cloned_audio_path, language=self.target_language
                )
            else:
                self.voice_cloner.clone_voice(
                    translated_text, segment['audio_path'], cloned_audio_path, language=self.target_language
                )
            
            # Update segment with processed data
            processed_segment = segment.copy()
            processed_segment['translated_text'] = translated_text
            processed_segment['cloned_audio_path'] = cloned_audio_path
            processed_segments.append(processed_segment)
        
        return processed_segments
    
    def _get_reference_audio(self, segment: Dict, all_segments: List[Dict], audio_path: str) -> Optional[str]:
        """Get reference audio for voice cloning from the same speaker"""
        speaker = segment.get('speaker', 'speaker_0')
        
        # Find other segments from the same speaker
        same_speaker_segments = [s for s in all_segments if s.get('speaker') == speaker]
        
        if len(same_speaker_segments) > 1:
            # Use the longest segment from the same speaker as reference
            longest_segment = max(same_speaker_segments, key=lambda s: s['end'] - s['start'])
            if longest_segment != segment:
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
    
    def _reconstruct_audio_unified(self, voice_segments: List[Dict], non_voice_segments: List[Dict], output_path: str):
        """Reconstruct audio with unified format handling"""
        print("🔧 Reconstructing audio with unified format...")
        
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
        # Extract original audio properties
        probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', input_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        input_info = json.loads(result.stdout)
        
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
        print("Usage: python multi_speaker_translator_v5.py <input_audio> <output_audio>")
        sys.exit(1)
    
    input_audio = sys.argv[1]
    output_audio = sys.argv[2]
    
    translator = MultiSpeakerTranslatV5()
    translator.translate_audio(input_audio, output_audio)

if __name__ == "__main__":
    main() 