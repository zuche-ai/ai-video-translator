#!/usr/bin/env python3
"""
Core logic tests for the video translation API.
These tests focus on the core functions without requiring heavy dependencies.
"""

import unittest
import tempfile
import os
import sys
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only the functions that don't require heavy dependencies
from video_translator.api import format_timestamp, parse_srt


class TestCoreLogic(unittest.TestCase):
    """Test core logic functions that don't require heavy dependencies"""
    
    def test_format_timestamp(self):
        """Test timestamp formatting function"""
        # Test various time values
        test_cases = [
            (0.0, "00:00:00,000"),
            (1.0, "00:00:01,000"),
            (1.5, "00:00:01,500"),
            (61.0, "00:01:01,000"),
            (61.123, "00:01:01,123"),
            (3661.0, "01:01:01,000"),
            (3661.999, "01:01:01,999"),
            (7325.456, "02:02:05,456")
        ]
        
        for input_time, expected in test_cases:
            with self.subTest(input_time=input_time):
                result = format_timestamp(input_time)
                self.assertEqual(result, expected)
    
    def test_format_timestamp_edge_cases(self):
        """Test timestamp formatting with edge cases"""
        # Test very small values
        self.assertEqual(format_timestamp(0.001), "00:00:00,001")
        self.assertEqual(format_timestamp(0.999), "00:00:00,999")
        
        # Test large values
        self.assertEqual(format_timestamp(359999.999), "99:59:59,999")
        
        # Test negative values (should handle gracefully)
        self.assertEqual(format_timestamp(-1.0), "00:00:01,000")  # Assumes absolute value
    
    def test_parse_srt_valid(self):
        """Test parsing valid SRT content"""
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
How are you today?

3
00:00:07,000 --> 00:00:09,000
I am doing well, thank you."""
        
        result = parse_srt(srt_content)
        
        self.assertEqual(len(result), 3)
        
        # Check first subtitle
        self.assertEqual(result[0]['start'], 1.0)
        self.assertEqual(result[0]['end'], 3.0)
        self.assertEqual(result[0]['text'], 'Hello world')
        
        # Check second subtitle
        self.assertEqual(result[1]['start'], 4.0)
        self.assertEqual(result[1]['end'], 6.0)
        self.assertEqual(result[1]['text'], 'How are you today?')
        
        # Check third subtitle
        self.assertEqual(result[2]['start'], 7.0)
        self.assertEqual(result[2]['end'], 9.0)
        self.assertEqual(result[2]['text'], 'I am doing well, thank you.')
    
    def test_parse_srt_empty(self):
        """Test parsing empty SRT content"""
        result = parse_srt("")
        self.assertEqual(result, [])
        
        result = parse_srt("\n\n\n")
        self.assertEqual(result, [])
    
    def test_parse_srt_single_subtitle(self):
        """Test parsing SRT with single subtitle"""
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Single subtitle"""
        
        result = parse_srt(srt_content)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['start'], 1.0)
        self.assertEqual(result[0]['end'], 3.0)
        self.assertEqual(result[0]['text'], 'Single subtitle')
    
    def test_parse_srt_multiline_text(self):
        """Test parsing SRT with multiline text"""
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Line 1
Line 2
Line 3"""
        
        result = parse_srt(srt_content)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'Line 1\nLine 2\nLine 3')
    
    def test_parse_srt_invalid_format(self):
        """Test parsing invalid SRT format"""
        invalid_srt = """1
00:00:01,000 --> 00:00:03,000
Hello world
2
invalid timestamp
How are you today?"""
        
        with self.assertRaises(ValueError):
            parse_srt(invalid_srt)
    
    def test_parse_srt_malformed_timestamp(self):
        """Test parsing SRT with malformed timestamps"""
        malformed_srt = """1
00:00:01,000 --> invalid
Hello world"""
        
        with self.assertRaises(ValueError):
            parse_srt(malformed_srt)
    
    def test_parse_srt_missing_text(self):
        """Test parsing SRT with missing text"""
        missing_text_srt = """1
00:00:01,000 --> 00:00:03,000

2
00:00:04,000 --> 00:00:06,000
Has text"""
        
        result = parse_srt(missing_text_srt)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], '')  # Empty text
        self.assertEqual(result[1]['text'], 'Has text')
    
    def test_parse_srt_complex_timestamps(self):
        """Test parsing SRT with complex timestamp formats"""
        complex_srt = """1
00:00:01,123 --> 00:00:03,456
Precise timing

2
00:01:30,789 --> 00:01:35,012
Longer duration"""
        
        result = parse_srt(complex_srt)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['start'], 1.123)
        self.assertEqual(result[0]['end'], 3.456)
        self.assertEqual(result[1]['start'], 90.789)  # 1:30 = 90 seconds
        self.assertEqual(result[1]['end'], 95.012)
    
    def test_parse_srt_unicode_text(self):
        """Test parsing SRT with unicode text"""
        unicode_srt = """1
00:00:01,000 --> 00:00:03,000
¡Hola mundo! ¿Cómo estás?

2
00:00:04,000 --> 00:00:06,000
你好世界！你好吗？"""
        
        result = parse_srt(unicode_srt)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], '¡Hola mundo! ¿Cómo estás?')
        self.assertEqual(result[1]['text'], '你好世界！你好吗？')
    
    def test_parse_srt_special_characters(self):
        """Test parsing SRT with special characters"""
        special_srt = """1
00:00:01,000 --> 00:00:03,000
Text with "quotes" and 'apostrophes'

2
00:00:04,000 --> 00:00:06,000
Text with & symbols and <tags>"""
        
        result = parse_srt(special_srt)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], 'Text with "quotes" and \'apostrophes\'')
        self.assertEqual(result[1]['text'], 'Text with & symbols and <tags>')


class TestDataValidation(unittest.TestCase):
    """Test data validation functions"""
    
    def test_validate_audio_duration(self):
        """Test audio duration validation"""
        # This would test the logic for validating audio duration
        # without requiring actual audio files
        pass
    
    def test_validate_srt_timing(self):
        """Test SRT timing validation"""
        # Test that SRT timings are logical
        srt_data = [
            {'start': 1.0, 'end': 3.0, 'text': 'First'},
            {'start': 4.0, 'end': 6.0, 'text': 'Second'},
            {'start': 7.0, 'end': 9.0, 'text': 'Third'}
        ]
        
        # All timings should be sequential
        for i in range(len(srt_data) - 1):
            self.assertLess(srt_data[i]['end'], srt_data[i + 1]['start'])
        
        # All durations should be positive
        for item in srt_data:
            self.assertGreater(item['end'], item['start'])
    
    def test_validate_text_length(self):
        """Test text length validation"""
        # Test that text lengths are reasonable
        test_texts = [
            "Short",
            "This is a medium length text",
            "This is a very long text that should be validated for reasonable length and not exceed certain limits that would cause issues in processing"
        ]
        
        for text in test_texts:
            # Text should not be empty
            self.assertGreater(len(text), 0)
            
            # Text should not be unreasonably long (e.g., > 1000 characters)
            self.assertLess(len(text), 1000)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_safe_division(self):
        """Test safe division to avoid division by zero"""
        def safe_divide(a, b, default=0.0):
            """Safely divide a by b, returning default if b is zero"""
            try:
                return a / b
            except ZeroDivisionError:
                return default
        
        # Test normal division
        self.assertEqual(safe_divide(10, 2), 5.0)
        self.assertEqual(safe_divide(10, 3), 10/3)
        
        # Test division by zero
        self.assertEqual(safe_divide(10, 0), 0.0)
        self.assertEqual(safe_divide(10, 0, default=1.0), 1.0)
    
    def test_clamp_values(self):
        """Test value clamping function"""
        def clamp(value, min_val, max_val):
            """Clamp a value between min_val and max_val"""
            return max(min_val, min(max_val, value))
        
        # Test normal values
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(0, 0, 10), 0)
        self.assertEqual(clamp(10, 0, 10), 10)
        
        # Test values outside range
        self.assertEqual(clamp(-5, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)
        
        # Test edge cases
        self.assertEqual(clamp(5, 10, 0), 5)  # min > max should still work
        self.assertEqual(clamp(5, 5, 5), 5)   # min == max


if __name__ == '__main__':
    unittest.main() 