"""
LDS Glossary Loader

This module handles loading and managing LDS-specific terms and their translations
from a CSV file for post-translation replacement.
"""

import csv
import os
import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class LDSGlossaryLoader:
    """Loads and manages LDS-specific terms for translation enhancement."""
    
    def __init__(self, glossary_path: str = "lds_glossary.csv"):
        """
        Initialize the glossary loader.
        
        Args:
            glossary_path: Path to the CSV file containing LDS terms
        """
        self.glossary_path = glossary_path
        self.glossary = {}
        self.case_sensitive_terms = {}
        self.load_glossary()
    
    def load_glossary(self) -> None:
        """Load the glossary from the CSV file."""
        if not os.path.exists(self.glossary_path):
            logger.warning(f"Glossary file not found: {self.glossary_path}")
            return
        
        try:
            with open(self.glossary_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    original = row.get('original_term', '').strip()
                    translated = row.get('translated_term', '').strip()
                    context = row.get('context_notes', '').strip()
                    
                    if original and translated:
                        # Store both case-sensitive and case-insensitive versions
                        self.glossary[original.lower()] = {
                            'original': original,
                            'translated': translated,
                            'context': context
                        }
                        
                        # Also store case-sensitive version for exact matches
                        self.case_sensitive_terms[original] = {
                            'original': original,
                            'translated': translated,
                            'context': context
                        }
            
            logger.info(f"Loaded {len(self.glossary)} LDS terms from glossary")
            
        except Exception as e:
            logger.error(f"Error loading glossary: {e}")
    
    def get_translation(self, term: str) -> str:
        """
        Get the translation for a given term.
        
        Args:
            term: The term to translate
            
        Returns:
            The translated term, or the original if no translation found
        """
        # First try exact case-sensitive match
        if term in self.case_sensitive_terms:
            return self.case_sensitive_terms[term]['translated']
        
        # Then try case-insensitive match
        if term.lower() in self.glossary:
            return self.glossary[term.lower()]['translated']
        
        return term
    
    def apply_glossary_replacements(self, text: str) -> str:
        """
        Apply glossary replacements to translated text.
        
        Args:
            text: The translated text to process
            
        Returns:
            Text with LDS terms properly replaced
        """
        if not self.glossary:
            return text
        
        # Sort terms by length (longest first) to avoid partial replacements
        sorted_terms = sorted(
            self.case_sensitive_terms.keys(),
            key=len,
            reverse=True
        )
        
        processed_text = text
        
        for original_term in sorted_terms:
            translation = self.case_sensitive_terms[original_term]['translated']
            
            # Use word boundaries to avoid partial word replacements
            pattern = r'\b' + re.escape(original_term) + r'\b'
            processed_text = re.sub(pattern, translation, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def get_glossary_stats(self) -> Dict[str, int]:
        """
        Get statistics about the loaded glossary.
        
        Returns:
            Dictionary with glossary statistics
        """
        return {
            'total_terms': len(self.glossary),
            'case_sensitive_terms': len(self.case_sensitive_terms)
        }
    
    def list_terms(self) -> List[Dict[str, str]]:
        """
        Get a list of all terms in the glossary.
        
        Returns:
            List of dictionaries containing term information
        """
        return list(self.case_sensitive_terms.values())
    
    def add_term(self, original: str, translated: str, context: str = "") -> None:
        """
        Add a new term to the glossary.
        
        Args:
            original: Original English term
            translated: Translated term
            context: Context notes
        """
        self.glossary[original.lower()] = {
            'original': original,
            'translated': translated,
            'context': context
        }
        self.case_sensitive_terms[original] = {
            'original': original,
            'translated': translated,
            'context': context
        }
        logger.info(f"Added new term: {original} -> {translated}")
    
    def save_glossary(self) -> None:
        """Save the current glossary back to the CSV file."""
        try:
            with open(self.glossary_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['original_term', 'translated_term', 'context_notes'])
                writer.writeheader()
                
                for term_info in self.case_sensitive_terms.values():
                    writer.writerow({
                        'original_term': term_info['original'],
                        'translated_term': term_info['translated'],
                        'context_notes': term_info['context']
                    })
            
            logger.info(f"Saved glossary to {self.glossary_path}")
            
        except Exception as e:
            logger.error(f"Error saving glossary: {e}")


# Global instance for easy access
_glossary_loader = None

def get_glossary_loader() -> LDSGlossaryLoader:
    """
    Get the global glossary loader instance.
    
    Returns:
        The LDSGlossaryLoader instance
    """
    global _glossary_loader
    if _glossary_loader is None:
        _glossary_loader = LDSGlossaryLoader()
    return _glossary_loader

def apply_lds_glossary(text: str) -> str:
    """
    Apply LDS glossary replacements to text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with LDS terms replaced
    """
    loader = get_glossary_loader()
    return loader.apply_glossary_replacements(text) 