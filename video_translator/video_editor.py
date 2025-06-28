import ffmpeg
import os
from typing import Optional

def burn_subtitles(video_path: str, srt_path: str, output_path: str, debug: bool = False) -> None:
    """
    Burn subtitles into video using ffmpeg, preserving the original audio.
    
    Args:
        video_path: Path to the input video file
        srt_path: Path to the SRT subtitle file
        output_path: Path for the output video file
        debug: Enable debug output
        
    Raises:
        FileNotFoundError: If video or SRT file doesn't exist
        ImportError: If ffmpeg-python is not installed
        RuntimeError: If video processing fails
    """
    # Check if input files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
    
    try:
        if debug:
            print(f"[DEBUG] Burning subtitles from {srt_path} into {video_path} (preserving audio)...")
        
        # Input video
        input_stream = ffmpeg.input(video_path)
        # Video with subtitles
        video = input_stream.video.filter('subtitles', srt_path)
        # Audio (copy original)
        audio = input_stream.audio
        # Output both
        stream = ffmpeg.output(video, audio, output_path, vcodec='libx264', acodec='copy')
        
        if debug:
            print(f"[DEBUG] Running ffmpeg command...")
        
        ffmpeg.run(stream, overwrite_output=True, quiet=not debug)
        
        if debug:
            print(f"[DEBUG] Video processing completed. Output saved to {output_path}")
            
    except ImportError as e:
        raise ImportError("Ffmpeg-python is not installed. Please run: pip install ffmpeg-python") from e
    except Exception as e:
        raise RuntimeError(f"Video processing failed: {str(e)}") from e
