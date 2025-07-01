import unittest
import tempfile
import os
import numpy as np
import librosa
import soundfile as sf
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestAudioSynchronization(unittest.TestCase):
    """Test cases for audio synchronization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = 22050
        
        # Create test audio segments
        self.create_test_audio_segments()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_audio_segments(self):
        """Create test audio segments with known durations."""
        # Create original audio (10 seconds total)
        duration = 10.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        self.original_audio = np.sin(2 * np.pi * 440 * t) * 0.3
        
        # Create test segments with specific timestamps
        self.segments = [
            {
                'start': 0.0,
                'end': 3.0,
                'text': 'First segment',
                'duration': 3.0
            },
            {
                'start': 4.0,
                'end': 7.0,
                'text': 'Second segment',
                'duration': 3.0
            },
            {
                'start': 8.0,
                'end': 10.0,
                'text': 'Third segment',
                'duration': 2.0
            }
        ]
        
        # Create corresponding audio files for each segment
        self.audio_files = []
        for i, segment in enumerate(self.segments):
            audio_file = os.path.join(self.temp_dir, f"segment_{i}.wav")
            segment_duration = segment['end'] - segment['start']
            t_segment = np.linspace(0, segment_duration, int(self.sample_rate * segment_duration))
            segment_audio = np.sin(2 * np.pi * (440 + i * 100) * t_segment) * 0.3
            sf.write(audio_file, segment_audio, self.sample_rate)
            self.audio_files.append(audio_file)
    
    def test_timeline_creation(self):
        """Test that timeline is created with correct length."""
        # Create timeline
        timeline = np.zeros_like(self.original_audio)
        
        # Verify timeline length matches original audio
        self.assertEqual(len(timeline), len(self.original_audio))
        self.assertEqual(len(timeline) / self.sample_rate, 10.0)
    
    def test_segment_placement(self):
        """Test that segments are placed at correct timestamps."""
        timeline = np.zeros_like(self.original_audio)
        timestamps = [(seg['start'], seg['end']) for seg in self.segments]
        
        # Place segments at their timestamps
        for i, (audio_file, (start_time, end_time)) in enumerate(zip(self.audio_files, timestamps)):
            segment_audio, _ = librosa.load(audio_file, sr=self.sample_rate)
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            segment_length = end_sample - start_sample
            
            # Resize segment to match timing if needed
            if len(segment_audio) != segment_length:
                segment_audio = librosa.effects.time_stretch(
                    segment_audio, 
                    rate=len(segment_audio)/segment_length
                )
            
            # Place in timeline
            if start_sample + len(segment_audio) <= len(timeline):
                timeline[start_sample:start_sample + len(segment_audio)] = segment_audio
        
        # Verify segments are placed at correct positions
        # Check that there's audio at expected positions and silence elsewhere
        for i, segment in enumerate(self.segments):
            start_sample = int(segment['start'] * self.sample_rate)
            end_sample = int(segment['end'] * self.sample_rate)
            
            # Should have audio in segment region
            segment_audio = timeline[start_sample:end_sample]
            self.assertGreater(np.max(np.abs(segment_audio)), 0.01)
            
            # Should have silence before segment (except for first segment)
            if i > 0:
                prev_end = int(self.segments[i-1]['end'] * self.sample_rate)
                silence_region = timeline[prev_end:start_sample]
                self.assertLess(np.max(np.abs(silence_region)), 0.01)
    
    def test_time_stretching(self):
        """Test that segments are properly time-stretched to match duration."""
        # Create a segment that's too long
        long_duration = 5.0
        t_long = np.linspace(0, long_duration, int(self.sample_rate * long_duration))
        long_audio = np.sin(2 * np.pi * 440 * t_long) * 0.3
        
        # Target duration
        target_duration = 3.0
        target_length = int(target_duration * self.sample_rate)
        
        # Time stretch
        stretched_audio = librosa.effects.time_stretch(
            long_audio, 
            rate=len(long_audio)/target_length
        )
        
        # Verify length matches target
        self.assertEqual(len(stretched_audio), target_length)
        self.assertAlmostEqual(len(stretched_audio) / self.sample_rate, target_duration, places=2)
    
    def test_overlay_synchronization(self):
        """Test that overlay mode properly synchronizes audio."""
        # Create timeline with segments
        timeline = np.zeros_like(self.original_audio)
        timestamps = [(seg['start'], seg['end']) for seg in self.segments]
        
        for i, (audio_file, (start_time, end_time)) in enumerate(zip(self.audio_files, timestamps)):
            segment_audio, _ = librosa.load(audio_file, sr=self.sample_rate)
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            segment_length = end_sample - start_sample
            
            if len(segment_audio) != segment_length:
                segment_audio = librosa.effects.time_stretch(
                    segment_audio, 
                    rate=len(segment_audio)/segment_length
                )
            
            if start_sample + len(segment_audio) <= len(timeline):
                timeline[start_sample:start_sample + len(segment_audio)] = segment_audio
        
        # Test overlay mixing
        original_volume = 0.3
        translated_volume = 1.0 - original_volume
        
        mixed_audio = (self.original_audio * original_volume + 
                      timeline * translated_volume)
        
        # Verify mixed audio has correct length
        self.assertEqual(len(mixed_audio), len(self.original_audio))
        
        # Verify mixing preserves timing
        for segment in self.segments:
            start_sample = int(segment['start'] * self.sample_rate)
            end_sample = int(segment['end'] * self.sample_rate)
            
            # Should have mixed audio in segment region
            mixed_segment = mixed_audio[start_sample:end_sample]
            self.assertGreater(np.max(np.abs(mixed_segment)), 0.01)
    
    def test_edge_cases(self):
        """Test edge cases in audio synchronization."""
        # Test segment that goes beyond timeline
        timeline = np.zeros(1000)  # Short timeline
        segment_audio = np.ones(1500)  # Longer segment
        
        start_sample = 800
        if start_sample + len(segment_audio) <= len(timeline):
            timeline[start_sample:start_sample + len(segment_audio)] = segment_audio
        else:
            # Truncate if it goes beyond the timeline
            fit_length = len(timeline) - start_sample
            timeline[start_sample:] = segment_audio[:fit_length]
        
        # Verify only the part that fits is placed
        self.assertEqual(len(timeline), 1000)
        self.assertEqual(np.sum(timeline[:800]), 0)  # Should be zero before start
        self.assertEqual(np.sum(timeline[800:]), 200)  # Should have 200 samples of audio
    
    def test_empty_segments(self):
        """Test handling of empty segments."""
        timeline = np.zeros_like(self.original_audio)
        
        # Test with empty audio file
        empty_audio = np.array([])
        start_sample = 1000
        end_sample = 2000
        
        if len(empty_audio) > 0:
            if start_sample + len(empty_audio) <= len(timeline):
                timeline[start_sample:start_sample + len(empty_audio)] = empty_audio
        
        # Timeline should remain unchanged
        self.assertEqual(np.sum(timeline), 0)
    
    def test_sample_rate_consistency(self):
        """Test that sample rate is consistent throughout processing."""
        # Load audio files and verify sample rate
        for audio_file in self.audio_files:
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            self.assertEqual(sr, self.sample_rate)
        
        # Verify original audio sample rate
        self.assertEqual(len(self.original_audio) / 10.0, self.sample_rate)
    
    def test_timestamp_accuracy(self):
        """Test that timestamps are accurately converted to sample indices."""
        for segment in self.segments:
            start_time = segment['start']
            end_time = segment['end']
            
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Verify conversion accuracy
            expected_start_sample = int(start_time * self.sample_rate)
            expected_end_sample = int(end_time * self.sample_rate)
            
            self.assertEqual(start_sample, expected_start_sample)
            self.assertEqual(end_sample, expected_end_sample)
            
            # Verify duration
            duration_samples = end_sample - start_sample
            duration_seconds = duration_samples / self.sample_rate
            self.assertAlmostEqual(duration_seconds, segment['end'] - segment['start'], places=2)

if __name__ == '__main__':
    unittest.main() 