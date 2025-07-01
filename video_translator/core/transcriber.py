import whisper
import os
import warnings
import torch
import time
from typing import List, Dict, Any

def transcribe_video(video_path: str, language: str = "en", debug: bool = False, timeout: int = 300) -> List[Dict[str, Any]]:
    """
    Transcribe video audio using Whisper.
    
    Args:
        video_path: Path to the video file
        language: Source language code (e.g., 'en', 'es', 'fr')
        debug: Enable debug output
        timeout: Timeout in seconds for transcription (not implemented yet)
        
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
        print(f"[TRANSCRIBE] Starting transcription process...")
        print(f"[TRANSCRIBE] Video path: {video_path}")
        print(f"[TRANSCRIBE] Language: {language}")
        
        # Detect if we're running on CPU and suppress FP16 warning
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TRANSCRIBE] Using device: {device}")
        
        if device == "cpu":
            # Suppress the FP16 warning on CPU since user can't do anything about it
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
                
                # Load Whisper model
                print(f"[TRANSCRIBE] Loading Whisper model...")
                model = whisper.load_model("base")
                print(f"[TRANSCRIBE] Whisper model loaded successfully")
                
                if debug:
                    print(f"[DEBUG] Transcribing {video_path} in {language}...")
                
                # Transcribe the video (also within the warning suppression context)
                print(f"[TRANSCRIBE] Starting transcription...")
                result = model.transcribe(video_path, language=language)
                print(f"[TRANSCRIBE] Transcription completed successfully")
        else:
            # GPU available, load model normally
            print(f"[TRANSCRIBE] Loading Whisper model...")
            model = whisper.load_model("base")
            print(f"[TRANSCRIBE] Whisper model loaded successfully")
            
            if debug:
                print(f"[DEBUG] Transcribing {video_path} in {language}...")
            
            # Transcribe the video
            print(f"[TRANSCRIBE] Starting transcription...")
            result = model.transcribe(video_path, language=language)
            print(f"[TRANSCRIBE] Transcription completed successfully")
        
        if debug:
            print(f"[DEBUG] Transcription completed. Found {len(result['segments'])} segments.")
        
        print(f"[TRANSCRIBE] Returning {len(result['segments'])} segments")
        return result['segments']
        
    except ImportError as e:
        raise ImportError("Whisper is not installed. Please run: pip install openai-whisper") from e
    except Exception as e:
        print(f"[TRANSCRIBE] Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Transcription failed: {str(e)}") from e
