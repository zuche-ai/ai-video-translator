import argostranslate.package
import argostranslate.translate
from typing import List, Dict, Any

def translate_segments(segments: List[Dict[str, Any]], src_lang: str, tgt_lang: str, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Translate text segments from source language to target language.
    
    Args:
        segments: List of segments with 'text' field
        src_lang: Source language code (e.g., 'en', 'es', 'fr')
        tgt_lang: Target language code
        debug: Enable debug output
        
    Returns:
        List of segments with translated text
        
    Raises:
        ImportError: If argostranslate is not installed
        ValueError: If segments list is empty or invalid
        RuntimeError: If translation fails
    """
    if not segments:
        raise ValueError("No segments provided for translation")
    
    try:
        if debug:
            print(f"[DEBUG] Installing translation package for {src_lang} to {tgt_lang}...")
        
        # Install translation package if not already installed
        try:
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                (x for x in available_packages if x.from_code == src_lang and x.to_code == tgt_lang), None
            )
            if package_to_install:
                argostranslate.package.install_from_path(package_to_install.download())
        except Exception as e:
            if debug:
                print(f"[DEBUG] Warning: Could not install translation package: {e}")
        
        if debug:
            print(f"[DEBUG] Translating {len(segments)} segments...")
        
        # Translate each segment
        translated_segments = []
        for i, segment in enumerate(segments):
            if 'text' not in segment:
                raise ValueError(f"Segment {i} missing 'text' field")
            
            translated_text = argostranslate.translate.translate(segment['text'], src_lang, tgt_lang)
            translated_segment = segment.copy()
            translated_segment['text'] = translated_text
            translated_segments.append(translated_segment)
            
            if debug:
                print(f"[DEBUG] '{segment['text'][:50]}...' -> '{translated_text[:50]}...'")
        
        if debug:
            print(f"[DEBUG] Translation completed.")
        
        return translated_segments
        
    except ImportError as e:
        raise ImportError("Argostranslate is not installed. Please run: pip install argostranslate") from e
    except ValueError:
        # Re-raise ValueError without wrapping
        raise
    except Exception as e:
        raise RuntimeError(f"Translation failed: {str(e)}") from e
