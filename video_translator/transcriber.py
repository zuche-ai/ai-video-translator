import whisper
import os
from typing import List, Dict, Any

def transcribe_video(video_path: str, language: str = "en", debug: bool = False) -> List[Dict[str, Any]]:
    """
    Transcribe video audio using Whisper.
    
    Args:
        video_path: Path to the video file
        language: Source language code (e.g., 'en', 'es', 'fr')
        debug: Enable debug output
        
    Returns:
        List of segments with start, end, and text
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ImportError: If whisper is not installed
        RuntimeError: If transcription fails
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        if debug:
            print(f"[DEBUG] Loading Whisper model...")
        
        # Load Whisper model
        model = whisper.load_model("base")
        
        if debug:
            print(f"[DEBUG] Transcribing {video_path} in {language}...")
        
        # Transcribe the video
        result = model.transcribe(video_path, language=language)
        
        if debug:
            print(f"[DEBUG] Transcription completed. Found {len(result['segments'])} segments.")
        
        return result['segments']
        
    except ImportError as e:
        raise ImportError("Whisper is not installed. Please run: pip install openai-whisper") from e
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}") from e
