import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_translator.core.translator import translate_segments


class TestTranslator(unittest.TestCase):
    
    def setUp(self):
        self.test_segments = [
            {'start': 0.0, 'end': 5.0, 'text': 'Hello world'},
            {'start': 5.0, 'end': 10.0, 'text': 'How are you?'}
        ]
    
    @patch('video_translator.core.translator.argostranslate.package')
    @patch('video_translator.core.translator.argostranslate.translate')
    def test_translate_segments_success(self, mock_translate, mock_package):
        """Test successful translation"""
        # Mock translation
        mock_translate.translate.side_effect = ['Hola mundo', '¿Cómo estás?']
        
        # Test the function
        result = translate_segments(self.test_segments, 'en', 'es', debug=False)
        
        # Verify results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], 'Hola mundo')
        self.assertEqual(result[1]['text'], '¿Cómo estás?')
        self.assertEqual(result[0]['start'], 0.0)  # Original fields preserved
        self.assertEqual(result[0]['end'], 5.0)
    
    def test_translate_segments_empty_list(self):
        """Test error when segments list is empty"""
        with self.assertRaises(ValueError) as context:
            translate_segments([], 'en', 'es')
        
        self.assertIn("No segments provided", str(context.exception))
    
    def test_translate_segments_missing_text_field(self):
        """Test error when segment is missing text field"""
        invalid_segments = [
            {'start': 0.0, 'end': 5.0, 'text': 'Hello'},
            {'start': 5.0, 'end': 10.0}  # Missing text field
        ]
        
        with self.assertRaises(ValueError) as context:
            translate_segments(invalid_segments, 'en', 'es')
        
        self.assertIn("missing 'text' field", str(context.exception))
    
    @patch('video_translator.core.translator.argostranslate.package')
    @patch('video_translator.core.translator.argostranslate.translate')
    def test_translate_segments_translation_fails(self, mock_translate, mock_package):
        """Test error when translation fails"""
        mock_translate.translate.side_effect = Exception("Translation error")
        
        with self.assertRaises(RuntimeError) as context:
            translate_segments(self.test_segments, 'en', 'es')
        
        self.assertIn("Translation failed", str(context.exception))
    
    @patch('video_translator.core.translator.argostranslate.package')
    @patch('video_translator.core.translator.argostranslate.translate')
    def test_translate_segments_debug_mode(self, mock_translate, mock_package):
        """Test translation with debug mode enabled"""
        mock_translate.translate.side_effect = ['Hola mundo', '¿Cómo estás?']
        
        # Test with debug=True (should not raise any errors)
        result = translate_segments(self.test_segments, 'en', 'es', debug=True)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], 'Hola mundo')


if __name__ == '__main__':
    unittest.main() 