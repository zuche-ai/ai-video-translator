import ffmpeg
import os
from typing import Optional

def burn_subtitles(video_path: str, srt_path: Optional[str], output_path: str, 
                  audio_file: Optional[str] = None, caption_font_size: int = 24, debug: bool = False) -> None:
    """
    Burn subtitles into video using ffmpeg, with optional custom audio and font size.
    
    Args:
        video_path: Path to the input video file
        srt_path: Path to the SRT subtitle file (None for no subtitles)
        output_path: Path for the output video file
        audio_file: Optional path to custom audio file (if None, uses original audio)
        caption_font_size: Font size for subtitles (default: 24)
        debug: Enable debug output
        
    Raises:
        FileNotFoundError: If video, SRT, or audio file doesn't exist
        ImportError: If ffmpeg-python is not installed
        RuntimeError: If video processing fails
    """
    # Check if input files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if srt_path and not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
    if audio_file and not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    try:
        if debug:
            if srt_path and audio_file:
                print(f"[DEBUG] Burning subtitles from {srt_path} into {video_path} with custom audio {audio_file}...")
            elif srt_path:
                print(f"[DEBUG] Burning subtitles from {srt_path} into {video_path} (preserving original audio)...")
            elif audio_file:
                print(f"[DEBUG] Replacing audio in {video_path} with custom audio {audio_file} (no subtitles)...")
            else:
                print(f"[DEBUG] Copying {video_path} to {output_path} (no changes)...")
        
        # Input video
        input_stream = ffmpeg.input(video_path)
        
        # Video processing
        if srt_path:
            # Add subtitles with custom font size
            force_style = f"Fontsize={caption_font_size}"
            video = input_stream.video.filter('subtitles', srt_path, force_style=force_style)
        else:
            # No subtitles, use original video
            video = input_stream.video
        
        # Audio handling
        if audio_file:
            # Use custom audio file
            audio = ffmpeg.input(audio_file).audio
        else:
            # Use original audio
            audio = input_stream.audio
        
        # Output both
        stream = ffmpeg.output(video, audio, output_path, vcodec='libx264', acodec='aac')
        
        if debug:
            print(f"[DEBUG] Running ffmpeg command...")
        
        ffmpeg.run(stream, overwrite_output=True, quiet=not debug)
        
        if debug:
            print(f"[DEBUG] Video processing completed. Output saved to {output_path}")
            
    except ImportError as e:
        raise ImportError("Ffmpeg-python is not installed. Please run: pip install ffmpeg-python") from e
    except Exception as e:
        raise RuntimeError(f"Video processing failed: {str(e)}") from e
