import whisper
import os
import warnings
import torch
import time
from typing import List, Dict, Any

def transcribe_video(video_path: str, language: str = "en", debug: bool = False, timeout: int = 300, 
                    max_segment_length: float = 30.0, word_timestamps: bool = False) -> List[Dict[str, Any]]:
    """
    Transcribe video audio using Whisper.
    
    Args:
        video_path: Path to the video file
        language: Source language code (e.g., 'en', 'es', 'fr')
        debug: Enable debug output
        timeout: Timeout in seconds for transcription (not implemented yet)
        max_segment_length: Maximum segment length in seconds (default: 30.0)
        word_timestamps: Whether to include word-level timestamps (default: False)
        
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
        print(f"[TRANSCRIBE] Max segment length: {max_segment_length}s")
        print(f"[TRANSCRIBE] Word timestamps: {word_timestamps}")
        
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
                
                # Transcribe the video with custom options
                print(f"[TRANSCRIBE] Starting transcription...")
                result = model.transcribe(
                    video_path, 
                    language=language,
                    word_timestamps=word_timestamps,
                    condition_on_previous_text=True,  # Better continuity between segments
                    temperature=0.0,  # More deterministic output
                    compression_ratio_threshold=2.4,  # Less aggressive segmentation
                    logprob_threshold=-1.0,  # More lenient threshold
                    no_speech_threshold=0.6  # More lenient silence detection
                )
                print(f"[TRANSCRIBE] Transcription completed successfully")
        else:
            # GPU available, load model normally
            print(f"[TRANSCRIBE] Loading Whisper model...")
            model = whisper.load_model("base")
            print(f"[TRANSCRIBE] Whisper model loaded successfully")
            
            if debug:
                print(f"[DEBUG] Transcribing {video_path} in {language}...")
            
            # Transcribe the video with custom options
            print(f"[TRANSCRIBE] Starting transcription...")
            result = model.transcribe(
                video_path, 
                language=language,
                word_timestamps=word_timestamps,
                condition_on_previous_text=True,  # Better continuity between segments
                temperature=0.0,  # More deterministic output
                compression_ratio_threshold=2.4,  # Less aggressive segmentation
                logprob_threshold=-1.0,  # More lenient threshold
                no_speech_threshold=0.6  # More lenient silence detection
            )
            print(f"[TRANSCRIBE] Transcription completed successfully")
        
        # Post-process segments to merge very short ones
        segments = result['segments']
        if debug:
            print(f"[DEBUG] Original segments: {len(segments)}")
        
        # Merge segments that are too short (less than 1 seconds) with adjacent segments
        merged_segments = []
        i = 0
        while i < len(segments):
            current_seg = segments[i]
            duration = float(current_seg['end']) - float(current_seg['start'])
            
            if duration < 1.0 and i < len(segments) - 1:
                # Merge with next segment
                next_seg = segments[i + 1]
                merged_seg = {
                    'start': float(current_seg['start']),
                    'end': float(next_seg['end']),
                    'text': str(current_seg['text']).strip() + ' ' + str(next_seg['text']).strip()
                }
                merged_segments.append(merged_seg)
                i += 2  # Skip next segment since we merged it
                if debug:
                    next_duration = float(next_seg['end']) - float(next_seg['start'])
                    print(f"[DEBUG] Merged segments {i-2} and {i-1}: {duration:.2f}s + {next_duration:.2f}s")
            else:
                merged_segments.append(current_seg)
                i += 1
        
        if debug:
            print(f"[DEBUG] After merging: {len(merged_segments)} segments")
        
        print(f"[TRANSCRIBE] Returning {len(merged_segments)} segments")
        return merged_segments
        
    except ImportError as e:
        raise ImportError("Whisper is not installed. Please run: pip install openai-whisper") from e
    except Exception as e:
        print(f"[TRANSCRIBE] Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Transcription failed: {str(e)}") from e
