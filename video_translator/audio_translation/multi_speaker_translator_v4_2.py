#!/usr/bin/env python3
"""
Multi-Speaker Audio Translator V4.2
Enhanced voice detection for overlapping music/voice segments
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

class MultiSpeakerTranslatorV4_2:
    def __init__(self, 
                 source_language: str = "en",
                 target_language: str = "es",
                 model_name: str = "base",
                 voice_clone_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.source_language = source_language
        self.target_language = target_language
        self.model_name = model_name
        self.voice_clone_model = voice_clone_model
        self.device = device
        
        # Initialize components
        self.voice_cloner = VoiceCloner(model_name=voice_clone_model)
        
        # Enhanced voice detection parameters
        self.voice_threshold = 0.3  # Lower threshold for more sensitive voice detection
        self.music_threshold = 0.6  # Higher threshold to distinguish music from voice
        self.min_voice_duration = 0.5  # Minimum duration for voice segments
        self.overlap_tolerance = 0.1  # Tolerance for overlapping segments
        
    def detect_voice_segments_enhanced(self, audio_path: str) -> List[Dict]:
        """
        Enhanced voice detection that can handle overlapping music and voice
        """
        print("🔍 Enhanced voice detection with music/voice separation...")
        
        # Load audio for analysis
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        # Use Whisper for initial transcription with detailed segments
        print("📝 Getting detailed Whisper segments...")
        segments = transcribe_video(audio_path, self.source_language, word_timestamps=True)
        
        if not segments:
            print("⚠️ No segments found, treating entire audio as voice")
            return [{
                'start': 0.0,
                'end': duration,
                'text': '',
                'speaker': 'speaker_0',
                'type': 'voice'
            }]
        print(f"📊 Found {len(segments)} Whisper segments")
        
        # Enhanced voice detection with spectral analysis
        voice_segments = []
        
        for i, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment['end']
            text = segment.get('text', '').strip()
            
            if not text:
                continue
                
            # Extract audio segment for spectral analysis
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            if len(segment_audio) == 0:
                continue
            
            # Enhanced spectral analysis for voice vs music detection
            voice_confidence = self._analyze_voice_characteristics(segment_audio, int(sr))
            
            # Determine if this is primarily voice or music+voice
            if voice_confidence > self.voice_threshold:
                # This is a voice segment (with possible background music)
                voice_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'speaker': f'speaker_{i}',
                    'type': 'voice',
                    'voice_confidence': voice_confidence,
                    'has_background_music': voice_confidence < self.music_threshold
                })
                print(f"🎤 Voice segment {i+1}: {start_time:.2f}s - {end_time:.2f}s (confidence: {voice_confidence:.2f})")
            else:
                print(f"🎵 Music segment {i+1}: {start_time:.2f}s - {end_time:.2f}s (confidence: {voice_confidence:.2f})")
        
        return voice_segments
    
    def _analyze_voice_characteristics(self, audio_segment: np.ndarray, sr: int) -> float:
        """
        Enhanced analysis to distinguish voice from music
        """
        if len(audio_segment) == 0:
            return 0.0
        
        # Spectral features
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
        
        # Spectral centroid (voice typically has lower centroid than music)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
        
        # Spectral rolloff (voice has more energy in lower frequencies)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)
        
        # Zero crossing rate (voice has moderate ZCR, music can be higher)
        zcr = librosa.feature.zero_crossing_rate(audio_segment)
        
        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio_segment)
        harmonic_ratio = np.sum(np.abs(harmonic)) / (np.sum(np.abs(harmonic)) + np.sum(np.abs(percussive)) + 1e-8)
        
        # Voice characteristics scoring
        voice_score = 0.0
        
        # MFCC variance (voice has more consistent MFCC patterns)
        mfcc_variance = np.var(mfccs)
        if mfcc_variance < 100:  # Voice typically has lower MFCC variance
            voice_score += 0.3
        
        # Spectral centroid (voice is typically 1000-4000 Hz)
        avg_centroid = np.mean(spectral_centroid)
        if 1000 < avg_centroid < 4000:
            voice_score += 0.2
        
        # Harmonic ratio (voice is more harmonic than percussive)
        if harmonic_ratio > 0.6:
            voice_score += 0.2
        
        # Zero crossing rate (voice has moderate ZCR)
        avg_zcr = np.mean(zcr)
        if 0.05 < avg_zcr < 0.15:
            voice_score += 0.2
        
        # Duration factor (longer segments are more likely to be voice)
        duration = len(audio_segment) / sr
        if duration > 1.0:
            voice_score += 0.1
        
        return min(voice_score, 1.0)
    
    def extract_non_voice_segments(self, audio_path: str, voice_segments: List[Dict]) -> List[Dict]:
        """
        Extract non-voice segments (silence, music, etc.) with improved detection
        """
        print("🎵 Extracting non-voice segments...")
        
        # Get audio duration
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        non_voice_segments = []
        
        # Find gaps between voice segments
        current_time = 0.0
        
        for voice_seg in voice_segments:
            voice_start = voice_seg['start']
            
            # If there's a gap before this voice segment
            if voice_start > current_time + self.overlap_tolerance:
                gap_duration = voice_start - current_time
                
                if gap_duration >= self.min_voice_duration:
                    non_voice_segments.append({
                        'start': current_time,
                        'end': voice_start,
                        'type': 'non_voice',
                        'duration': gap_duration
                    })
                    print(f"🔇 Non-voice segment: {current_time:.2f}s - {voice_start:.2f}s ({gap_duration:.2f}s)")
            
            current_time = voice_seg['end']
        
        # Check if there's a non-voice segment at the end
        if current_time < duration - self.overlap_tolerance:
            end_duration = duration - current_time
            if end_duration >= self.min_voice_duration:
                non_voice_segments.append({
                    'start': current_time,
                    'end': duration,
                    'type': 'non_voice',
                    'duration': end_duration
                })
                print(f"🔇 End non-voice segment: {current_time:.2f}s - {duration:.2f}s ({end_duration:.2f}s)")
        
        return non_voice_segments
    
    def translate_audio(self, input_path: str, output_path: str, reference_audio_path: Optional[str] = None) -> bool:
        """
        Main translation pipeline with enhanced voice detection
        """
        try:
            print(f"🚀 Starting V4.2 translation: {input_path}")
            
            # Create output directory
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Enhanced voice detection
            voice_segments = self.detect_voice_segments_enhanced(input_path)
            
            if not voice_segments:
                print("⚠️ No voice segments detected, copying original audio")
                shutil.copy2(input_path, output_path)
                return True
            
            print(f"🎤 Found {len(voice_segments)} voice segments")
            
            # Step 2: Extract non-voice segments
            non_voice_segments = self.extract_non_voice_segments(input_path, voice_segments)
            print(f"🔇 Found {len(non_voice_segments)} non-voice segments")
            
            # Step 3: Process voice segments
            processed_voice_segments = []
            
            for i, segment in enumerate(voice_segments):
                print(f"\n🎤 Processing voice segment {i+1}/{len(voice_segments)}")
                print(f"   Text: {segment['text'][:100]}...")
                print(f"   Time: {segment['start']:.2f}s - {segment['end']:.2f}s")
                
                # Extract audio for this segment
                segment_audio_path = self._extract_audio_segment(
                    input_path, segment['start'], segment['end'], f"voice_segment_{i}"
                )
                
                # Translate text
                translated_text = translate_segments([segment], self.source_language, self.target_language)[0]['text']
                print(f"   Translated: {translated_text[:100]}...")
                
                # Voice clone with reference audio
                cloned_audio_path = f"temp_cloned_voice_{i}.wav"
                if reference_audio_path:
                    self.voice_cloner.clone_voice(
                        translated_text, reference_audio_path, cloned_audio_path, language=self.target_language
                    )
                else:
                    self.voice_cloner.clone_voice(
                        translated_text, segment_audio_path, cloned_audio_path, language=self.target_language
                    )
                
                if cloned_audio_path and os.path.exists(cloned_audio_path):
                    processed_voice_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'audio_path': cloned_audio_path,
                        'type': 'voice',
                        'original_text': segment['text'],
                        'translated_text': translated_text
                    })
                    print(f"   ✅ Voice cloned successfully")
                else:
                    print(f"   ❌ Voice cloning failed, using original audio")
                    processed_voice_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'audio_path': segment_audio_path,
                        'type': 'voice',
                        'original_text': segment['text'],
                        'translated_text': translated_text
                    })
            
            # Step 4: Extract non-voice audio segments
            non_voice_audio_segments = []
            
            for i, segment in enumerate(non_voice_segments):
                non_voice_audio_path = self._extract_audio_segment(
                    input_path, segment['start'], segment['end'], f"non_voice_segment_{i}"
                )
                non_voice_audio_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'audio_path': non_voice_audio_path,
                    'type': 'non_voice'
                })
            
            # Step 5: Reconstruct final audio with unified format
            self._reconstruct_audio_unified(
                processed_voice_segments, 
                non_voice_audio_segments, 
                output_path
            )
            
            print(f"\n✅ Translation complete: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Translation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_audio_segment(self, audio_path: str, start_time: float, end_time: float, segment_name: str) -> str:
        """Extract audio segment using ffmpeg"""
        output_path = f"temp_{segment_name}.wav"
        
        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c', 'copy',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def _reconstruct_audio_unified(self, voice_segments: List[Dict], non_voice_segments: List[Dict], output_path: str):
        """Reconstruct audio with unified format handling"""
        print("🔧 Reconstructing audio with unified format...")
        
        # Get original audio format from first non-voice segment (original audio)
        if non_voice_segments:
            probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', non_voice_segments[0]['audio_path']]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            original_format = json.loads(result.stdout)['format']
            print(f"📊 Original audio format: {original_format}")
        else:
            print("⚠️ No non-voice segments found, using default format")
            original_format = {'sample_rate': '16000', 'channels': '1'}
        
        # Convert voice segments to match original format
        converted_voice_segments = []
        for i, segment in enumerate(voice_segments):
            converted_path = f"temp_converted_voice_{i}.wav"
            
            # Convert voice segment to match original format
            convert_cmd = [
                'ffmpeg', '-y',
                '-i', segment['audio_path'],
                '-ar', original_format.get('sample_rate', '16000'),
                '-ac', original_format.get('channels', '1'),
                '-c:a', 'pcm_s16le',
                converted_path
            ]
            
            subprocess.run(convert_cmd, check=True, capture_output=True)
            print(f"🔄 Converted voice segment {i+1} to {original_format.get('sample_rate', '16000')}Hz")
            
            converted_voice_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'audio_path': converted_path,
                'type': 'voice',
                'original_text': segment['original_text'],
                'translated_text': segment['translated_text']
            })
        
        # Create file list for concatenation
        file_list_path = "temp_concat_list.txt"
        
        with open(file_list_path, 'w') as f:
            # Add all segments in chronological order
            all_segments = converted_voice_segments + non_voice_segments
            all_segments.sort(key=lambda x: x['start'])
            
            for segment in all_segments:
                f.write(f"file '{segment['audio_path']}'\n")
        
        # Concatenate with unified format
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list_path,
            '-c', 'copy',
            output_path
        ]
        
        subprocess.run(concat_cmd, check=True)
        
        # Cleanup
        os.remove(file_list_path)
        for segment in voice_segments + non_voice_segments:
            if os.path.exists(segment['audio_path']):
                os.remove(segment['audio_path'])
        for segment in converted_voice_segments:
            if os.path.exists(segment['audio_path']):
                os.remove(segment['audio_path'])

def main():
    """Main function for testing"""
    if len(sys.argv) < 3:
        print("Usage: python multi_speaker_translator_v4_2.py <input_audio> <output_audio> [reference_audio]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    reference_audio = sys.argv[3] if len(sys.argv) > 3 else None
    
    translator = MultiSpeakerTranslatorV4_2()
    success = translator.translate_audio(input_path, output_path, reference_audio)
    
    if success:
        print("🎉 Translation completed successfully!")
    else:
        print("❌ Translation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 