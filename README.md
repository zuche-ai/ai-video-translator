# Video Translator

A Python application that transcribes, translates, and adds subtitles to videos using AI-powered tools.

## Features

- **Audio Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
- **Translation**: Translates transcriptions using ArgosTranslate
- **Subtitle Generation**: Creates SRT subtitle files from translated text
- **Video Processing**: Burns subtitles into videos while preserving original audio
- **Error Handling**: Comprehensive error handling with helpful error messages
- **Unit Tests**: Full test coverage for all modules

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zuche-ai/ai-video-translator.git
   cd ai-video-translator
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg (required for video processing):**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

## Usage

### Basic Usage

```bash
python main.py --input video.mp4 --src-lang en --tgt-lang es --output translated_video.mp4
```

### Parameters

- `--input`: Path to the input video file
- `--src-lang`: Source language code (e.g., `en`, `fr`, `de`)
- `--tgt-lang`: Target language code for subtitles
- `--output`: Path for the output video file
- `--debug`: Enable debug output (optional)

### Language Codes

Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese

### Examples

```bash
# Translate English video to Spanish
python main.py --input my_video.mp4 --src-lang en --tgt-lang es --output spanish_video.mp4

# Translate French video to German with debug output
python main.py --input french_video.mp4 --src-lang fr --tgt-lang de --output german_video.mp4 --debug
```

## Error Handling

The application provides comprehensive error handling:

- **File Not Found**: Clear messages when input files don't exist
- **Missing Dependencies**: Helpful installation instructions for missing packages
- **Invalid Input**: Validation of input parameters and data
- **Processing Failures**: Detailed error messages for processing issues
- **Debug Mode**: Additional debugging information when `--debug` is used

### Common Error Solutions

1. **"ffmpeg not found"**: Install FFmpeg using your system's package manager
2. **"Whisper is not installed"**: Run `pip install openai-whisper`
3. **"Video file not found"**: Check the file path and ensure the video exists
4. **"No segments provided"**: The video may not contain speech or the audio is too quiet

## Testing

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run individual test modules
python -m unittest test_transcriber
python -m unittest test_translator
python -m unittest test_subtitles
python -m unittest test_video_editor
python -m unittest test_main
```

### Test Coverage

The test suite covers:

- **Transcriber Module**: Audio transcription functionality
- **Translator Module**: Text translation capabilities
- **Subtitles Module**: SRT file generation
- **Video Editor Module**: Subtitle burning and video processing
- **Main Module**: Command-line interface and error handling

Each module includes tests for:
- ✅ Successful operations
- ❌ Error conditions
- 🔍 Input validation
- 🐛 Debug mode functionality

## Project Structure

```
video_translator/
├── main.py              # Main application entry point
├── transcriber.py       # Audio transcription using Whisper
├── translator.py        # Text translation using ArgosTranslate
├── subtitles.py         # SRT subtitle file generation
├── video_editor.py      # Video processing with FFmpeg
├── requirements.txt     # Python dependencies
├── run_tests.py         # Test runner
├── test_*.py           # Unit test modules
└── README.md           # This file
```

## Dependencies

- **openai-whisper**: Audio transcription
- **argostranslate**: Text translation
- **pysrt**: Subtitle file handling
- **ffmpeg-python**: Video processing
- **tqdm**: Progress bars

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `python run_tests.py`
6. Submit a pull request

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
