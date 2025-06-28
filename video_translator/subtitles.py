import pysrt
import os
from typing import List, Dict, Any

def generate_srt(segments: List[Dict[str, Any]], video_path: str, language: str, debug: bool = False) -> str:
    """
    Generate SRT subtitle file from translated segments.
    
    Args:
        segments: List of segments with 'start', 'end', and 'text' fields
        video_path: Path to the original video file
        language: Target language code
        debug: Enable debug output
        
    Returns:
        Path to the generated SRT file
        
    Raises:
        ImportError: If pysrt is not installed
        ValueError: If segments list is empty or invalid
        RuntimeError: If SRT file creation fails
    """
    if not segments:
        raise ValueError("No segments provided for SRT generation")
    
    try:
        if debug:
            print(f"[DEBUG] Generating SRT file for {len(segments)} segments...")
        
        # Create subtitle objects
        subs = pysrt.SubRipFile()
        
        for i, segment in enumerate(segments):
            # Validate segment has required fields
            if 'start' not in segment or 'end' not in segment or 'text' not in segment:
                raise ValueError(f"Segment {i} missing required fields (start, end, text)")
            
            # Convert seconds to milliseconds
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            
            # Create subtitle item
            sub = pysrt.SubRipItem(
                index=i + 1,
                start=pysrt.SubRipTime(milliseconds=start_ms),
                end=pysrt.SubRipTime(milliseconds=end_ms),
                text=segment['text']
            )
            subs.append(sub)
            
            if debug:
                print(f"[DEBUG] Added subtitle {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        srt_path = f"{base_name}_{language}.srt"
        
        # Save SRT file
        subs.save(srt_path, encoding='utf-8')
        
        if debug:
            print(f"[DEBUG] SRT file saved to {srt_path}")
        
        return srt_path
        
    except ImportError as e:
        raise ImportError("Pysrt is not installed. Please run: pip install pysrt") from e
    except ValueError:
        # Re-raise ValueError without wrapping
        raise
    except Exception as e:
        raise RuntimeError(f"SRT file creation failed: {str(e)}") from e
