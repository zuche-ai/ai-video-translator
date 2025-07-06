# Video Translation Workflow Documentation

## Overview

This document describes the complete workflow for translating video/audio content from English to Spanish with voice cloning, preserving timing and non-speech events.

## Architecture

The system uses a **two-step process** to ensure quality and allow human review:

1. **Step 1**: Transcribe and translate → Generate SRT files for review
2. **Step 2**: Process with voice cloning → Generate final translated audio

## Step 1: Transcription and Translation

### Endpoint
```
POST /transcribe-and-translate
```

### Purpose
- Transcribe English audio to SRT format
- Translate English SRT to Spanish SRT
- Return both files for human review and editing

### Input
- Audio file (WAV, MP3, etc.)
- Source language (default: 'en')
- Target language (default: 'es')

### Output
- ZIP file containing:
  - `original_transcript.srt` - English transcription with timestamps
  - `translated_transcript.srt` - Spanish translation with same timestamps

### Example Usage
```bash
curl -X POST "http://localhost:5001/transcribe-and-translate" \
  -F "audio=@./christtestshort.wav" \
  -o ./results/transcripts.zip
```

### Processing Steps
1. **Audio Loading**: Convert input to 16kHz mono WAV
2. **Transcription**: Use Whisper model for English speech recognition
3. **Translation**: Use Helsinki-NLP translation model (en→es)
4. **SRT Generation**: Format with proper timestamps
5. **ZIP Creation**: Package both SRT files

## Step 2: Voice Cloning and Audio Generation

### Endpoint
```
POST /translate-audio-with-srt
```

### Purpose
- Take reviewed/edited SRT file
- Generate Spanish TTS with voice cloning
- Preserve original timing and non-speech events
- Output final translated audio

### Input
- Original audio file
- Reviewed/edited SRT file (Spanish translation)
- Voice cloning option (default: true)

### Output
- WAV file with Spanish speech and preserved timing

### Example Usage
```bash
curl -X POST "http://localhost:5001/translate-audio-with-srt" \
  -F "audio=@./christtestshort.wav" \
  -F "srt=@./results/translated_transcript.srt" \
  -o ./results/final_translated_audio.wav
```

### Processing Steps
1. **Audio Loading**: Load original audio at 16kHz
2. **SRT Parsing**: Extract speech segments and timing
3. **VAD Processing**: Identify non-speech segments using Silero VAD
4. **TTS Generation**: Generate Spanish speech for each segment using Coqui XTTS
5. **Voice Cloning**: Use original audio segments as voice reference
6. **Audio Splicing**: Combine TTS and non-speech segments with proper timing
7. **Time Stretching**: Adjust TTS duration to match original segment timing

## Complete Workflow Example

### 1. Generate SRT Files
```bash
# Start the transcription and translation process
curl -X POST "http://localhost:5001/transcribe-and-translate" \
  -F "audio=@./christtestshort.wav" \
  -o ./results/transcripts.zip

# Extract the SRT files
cd ./results
unzip transcripts.zip
```

### 2. Review and Edit SRT Files
- Open `original_transcript.srt` to review English transcription
- Open `translated_transcript.srt` to edit Spanish translation
- Make any timing adjustments if needed
- Save your changes

### 3. Generate Final Audio
```bash
# Use the edited SRT to generate final translated audio
curl -X POST "http://localhost:5001/translate-audio-with-srt" \
  -F "audio=@../christtestshort.wav" \
  -F "srt=@./translated_transcript.srt" \
  -o ./final_translated_audio.wav
```

## File Structure

```
video_translator/
├── christtestshort.wav                    # Input audio
├── results/
│   ├── transcripts.zip                    # Step 1 output
│   ├── original_transcript.srt            # English transcription
│   ├── translated_transcript.srt          # Spanish translation (editable)
│   └── final_translated_audio.wav         # Step 2 output
└── ...
```

## Key Features

### Voice Cloning
- Uses Coqui XTTS v2 for high-quality voice cloning
- Extracts reference audio from original speech segments
- Maintains speaker characteristics in translated speech

### Timing Preservation
- Preserves original segment timing from SRT
- Maintains non-speech events (silence, music, etc.)
- Uses time-stretching to match TTS duration to original timing

### Memory Efficiency
- Processes audio in chunks to handle large files
- Implements VAD to identify speech vs non-speech segments
- Optimizes TTS generation for memory usage

### Quality Control
- Human-in-the-loop review process
- Ability to edit translations and timing
- Two-step process ensures accuracy

## Technical Details

### Audio Processing
- **Sample Rate**: 16kHz for processing, 22.05kHz for TTS
- **Format**: WAV (PCM 16-bit)
- **Channels**: Mono

### Models Used
- **Transcription**: Whisper (base model)
- **Translation**: Helsinki-NLP/opus-mt-en-es
- **Voice Activity Detection**: Silero VAD
- **Text-to-Speech**: Coqui XTTS v2

### Speed Control
- Calculates TTS speed based on text length vs original duration
- Clamps speed between 0.5x and 2.0x to avoid artifacts
- Uses character-per-second estimation for Spanish

## Troubleshooting

### Common Issues

1. **Slow Motion Voice**
   - Check SRT segment durations vs actual speech
   - Verify speed calculation in TTS generation
   - Consider splitting long segments

2. **Memory Issues**
   - Large files may cause container crashes
   - System automatically skips trimming on very long files
   - Increase Docker memory allocation if needed

3. **TTS Quality**
   - Ensure reference audio segments are clear
   - Check that voice cloning is enabled
   - Verify Spanish text quality in SRT

### Debug Information
- Check Docker logs: `docker logs video-translator-backend`
- Look for debug messages with timing information
- Verify SRT file format and timing accuracy

## API Health Check

```bash
curl http://localhost:5001/health
```

Returns: `{"status": "healthy", "jobs": 0}`

## Dependencies

- **Docker**: Containerized environment
- **Python 3.11**: Runtime environment
- **ML Models**: Whisper, Helsinki-NLP, Silero VAD, Coqui XTTS
- **Audio Processing**: librosa, soundfile
- **Web Framework**: Flask

## Performance Notes

- **First Run**: Models are downloaded on first use (may take several minutes)
- **Processing Time**: Depends on audio length and complexity
- **Memory Usage**: ~8-16GB recommended for large files
- **Quality**: Best results with clear speech and good reference audio 