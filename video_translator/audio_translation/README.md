# Audio Translation Module

This module provides advanced audio translation capabilities with voice cloning and Voice Activity Detection (VAD).

## Features

- **Whisper Transcription**: High-quality speech-to-text using OpenAI's Whisper
- **Voice Activity Detection**: Automatically identifies voice and non-voice segments
- **Multi-language Translation**: Translate between multiple languages
- **Voice Cloning**: Generate translated audio that sounds like the original speaker
- **Intelligent Audio Stitching**: Preserve non-voice segments (music, sound effects) while translating speech

## Workflow

The audio translation process follows these steps:

1. **Transcribe Audio**: Convert speech to text using Whisper
2. **VAD Analysis**: Identify voice and non-voice segments using WebRTC VAD
3. **Translate Text**: Translate the transcription to the target language
4. **Adjust Timestamps**: Modify SRT timestamps to exclude non-voice segments
5. **Voice Cloning**: Generate voice-cloned audio for voice segments
6. **Extract Non-voice**: Preserve original audio for non-voice segments
7. **Stitch Audio**: Combine all segments into a continuous audio file

## Usage

### API Endpoint

```bash
POST /translate_audio
```

**Form Data:**
- `audio`: Input audio file (required)
- `reference_audio`: Reference audio for voice cloning (required)
- `src_lang`: Source language code (default: "en")
- `tgt_lang`: Target language code (default: "es")

**Response:**
```json
{
  "job_id": "uuid-string"
}
```

### Python API

```python
from video_translator.audio_translation.audio_translator import AudioTranslator

# Create translator
translator = AudioTranslator()

# Translate audio
output_path = translator.translate_audio(
    audio_path="input.wav",
    src_lang="en",
    tgt_lang="es",
    reference_audio_path="reference.wav",
    output_path="translated.wav"
)
```

### Convenience Function

```python
from video_translator.audio_translation.audio_translator import translate_audio_file

output_path = translate_audio_file(
    audio_path="input.wav",
    src_lang="en",
    tgt_lang="es",
    reference_audio_path="reference.wav"
)
```

## Progress Tracking

You can track progress using a callback function:

```python
def progress_callback(step, percent, status_message):
    print(f"Step {step}: {percent}% - {status_message}")

translator.translate_audio(
    audio_path="input.wav",
    src_lang="en",
    tgt_lang="es",
    reference_audio_path="reference.wav",
    progress_hook=progress_callback
)
```

## Supported Languages

The module supports all languages supported by:
- **Whisper**: For transcription
- **ArgosTranslate**: For text translation
- **XTTS**: For voice cloning

Common language codes:
- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `it`: Italian
- `pt`: Portuguese
- `ja`: Japanese
- `ko`: Korean
- `zh`: Chinese

## Audio Formats

**Input Formats:**
- WAV, MP3, M4A, FLAC, OGG, AAC, OPUS

**Output Format:**
- WAV (16-bit, 22.05kHz)

## Requirements

- `whisper`: Speech recognition
- `argostranslate`: Text translation
- `TTS`: Voice cloning (XTTS)
- `webrtcvad`: Voice activity detection
- `librosa`: Audio processing
- `ffmpeg`: Audio manipulation
- `numpy`: Numerical operations
- `soundfile`: Audio I/O

## Example

See `examples/audio_translation_example.py` for a complete working example.

## Testing

Run the test suite:

```bash
python -m pytest tests/audio_translation/test_audio_translator.py
```

## Troubleshooting

### Common Issues

1. **VAD Analysis Fails**: Ensure audio is 16kHz mono
2. **Voice Cloning Fails**: Check reference audio quality and length
3. **Translation Fails**: Verify language codes are supported
4. **Memory Issues**: Reduce audio file size or use smaller models

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance

- **Transcription**: ~1-2x real-time (depends on Whisper model)
- **Translation**: ~10-100x real-time (depends on text length)
- **Voice Cloning**: ~0.1-0.5x real-time (depends on XTTS model)
- **VAD Analysis**: ~10-50x real-time

## Memory Usage

- **Base Model**: ~1GB RAM
- **Large Model**: ~3GB RAM
- **XTTS Model**: ~2GB RAM
- **Total**: ~4-6GB RAM for full pipeline

## License

This module is part of the video_translator project and follows the same license terms. 