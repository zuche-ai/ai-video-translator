# Audio Translation Module

This module provides functionality to translate audio files from one language to another, with support for both regular text-to-speech (TTS) and voice cloning.

## Features

- **Audio Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
- **Text Translation**: Translates transcriptions using ArgosTranslate
- **Voice Cloning**: Clone the original speaker's voice for translated audio using Coqui XTTS
- **Regular TTS**: Generate translated audio using standard text-to-speech (placeholder for future implementation)
- **Multiple Audio Formats**: Supports MP3, WAV, M4A, FLAC, OGG, and AAC
- **Command-Line Interface**: Easy-to-use CLI for batch processing
- **Comprehensive Testing**: Full test coverage with unit and integration tests

## Quick Start

### Command-Line Usage

```bash
# Basic audio translation with voice cloning
python -m video_translator.audio_translation.cli \
  --input audio.mp3 \
  --src-lang en \
  --tgt-lang es \
  --voice-clone

# Specify custom output path
python -m video_translator.audio_translation.cli \
  --input audio.wav \
  --output translated_audio.wav \
  --src-lang en \
  --tgt-lang fr \
  --voice-clone

# Use different Whisper model size
python -m video_translator.audio_translation.cli \
  --input audio.mp3 \
  --src-lang en \
  --tgt-lang de \
  --model-size large \
  --voice-clone

# List supported audio formats
python -m video_translator.audio_translation.cli --list-formats
```

### Programmatic Usage

```python
from video_translator.audio_translation.audio_translator import AudioTranslator

# Create translator with voice cloning
translator = AudioTranslator(
    src_lang="en",
    tgt_lang="es",
    model_size="base",
    voice_clone=True
)

# Translate audio file
output_path = translator.translate_audio(
    input_path="input_audio.mp3",
    output_path="translated_audio.mp3"
)

print(f"Translation completed: {output_path}")
```

## Supported Languages

The module supports all languages available in:
- **Whisper**: For transcription (English, Spanish, French, German, etc.)
- **ArgosTranslate**: For text translation (100+ language pairs)
- **Coqui XTTS**: For voice cloning (multilingual support)

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)
- AAC (.aac)
- OPUS (.opus) - WhatsApp, Discord, WebRTC

## Architecture

The audio translation pipeline consists of three main steps:

1. **Transcription**: Convert audio to text using Whisper
2. **Translation**: Translate text using ArgosTranslate
3. **Audio Generation**: Generate new audio using voice cloning or TTS

```
Input Audio → Whisper → Text → ArgosTranslate → Translated Text → XTTS → Output Audio
```

## Configuration Options

### AudioTranslator Parameters

- `src_lang`: Source language code (default: "en")
- `tgt_lang`: Target language code (default: "es")
- `model_size`: Whisper model size - "tiny", "base", "small", "medium", "large" (default: "base")
- `voice_clone`: Whether to use voice cloning instead of regular TTS (default: False)

### CLI Options

- `--input, -i`: Input audio file path (required)
- `--output, -o`: Output audio file path (auto-generated if not specified)
- `--src-lang`: Source language code
- `--tgt-lang`: Target language code
- `--model-size`: Whisper model size
- `--voice-clone`: Use voice cloning
- `--verbose, -v`: Enable verbose logging
- `--list-formats`: List supported audio formats

## Examples

### Example 1: Basic Translation

```python
from video_translator.audio_translation.audio_translator import AudioTranslator

translator = AudioTranslator()
output_path = translator.translate_audio("english_audio.mp3")
# Output: english_audio_es.mp3
```

### Example 2: Custom Configuration

```python
translator = AudioTranslator(
    src_lang="fr",
    tgt_lang="de",
    model_size="large",
    voice_clone=True
)

output_path = translator.translate_audio(
    input_path="french_audio.wav",
    output_path="german_translation.wav"
)
```

### Example 3: Batch Processing

```python
import os
from pathlib import Path

translator = AudioTranslator(voice_clone=True)

input_dir = Path("input_audio")
output_dir = Path("translated_audio")
output_dir.mkdir(exist_ok=True)

for audio_file in input_dir.glob("*.mp3"):
    output_path = output_dir / f"translated_{audio_file.name}"
    translator.translate_audio(
        input_path=str(audio_file),
        output_path=str(output_path)
    )
```

## Error Handling

The module provides comprehensive error handling:

- **FileNotFoundError**: Input file doesn't exist
- **ValueError**: Unsupported audio format or empty transcription/translation
- **RuntimeError**: Transcription, translation, or audio generation failures
- **NotImplementedError**: Regular TTS not yet implemented (use voice cloning)

## Testing

Run the tests to ensure everything works correctly:

```bash
# Run all audio translation tests
pytest tests/audio_translation/

# Run with verbose output
pytest tests/audio_translation/ -v

# Run specific test
pytest tests/audio_translation/test_audio_translator.py::TestAudioTranslator::test_init_defaults
```

## Dependencies

The audio translation module requires:

- `openai-whisper`: For audio transcription
- `argostranslate`: For text translation
- `TTS` (Coqui): For voice cloning
- `torch`: For deep learning models
- `soundfile`: For audio file handling
- `librosa`: For audio processing

## Future Enhancements

- [ ] Implement regular TTS functionality
- [ ] Add support for more audio formats
- [ ] Implement batch processing optimization
- [ ] Add progress callbacks for long operations
- [ ] Support for custom voice models
- [ ] Integration with the web UI

## Troubleshooting

### Common Issues

1. **"Model not found" errors**: Ensure XTTS models are downloaded
2. **Memory issues**: Use smaller Whisper model sizes
3. **Slow processing**: Voice cloning is computationally intensive
4. **Audio quality**: Ensure input audio has good quality and clear speech

### Performance Tips

- Use "tiny" or "base" Whisper models for faster processing
- Ensure sufficient RAM (8GB+ recommended for voice cloning)
- Use SSD storage for better I/O performance
- Close other applications to free up system resources

## Contributing

When contributing to the audio translation module:

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation
4. Test with different audio formats and languages
5. Consider performance implications

## License

This module is part of the video translator project and follows the same license terms. 