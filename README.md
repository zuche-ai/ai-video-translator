# Video Translator

A Python application that transcribes, translates, and adds subtitles to videos using AI-powered tools. Now with **voice cloning** capabilities!

**Now with a modern local Web UI!** Use either the command-line interface (CLI) or the drag-and-drop React frontend for batch video translation, voice cloning, and subtitles—all running locally.

## Features

- **Audio Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
- **Translation**: Translates transcriptions using ArgosTranslate
- **Subtitle Generation**: Creates SRT subtitle files from translated text
- **Voice Cloning**: Clone the original speaker's voice for translated audio using Coqui TTS
- **Audio Overlay**: Replace or overlay translated audio on the original video
- **Video Processing**: Burns subtitles into videos with custom audio options
- **Error Handling**: Comprehensive error handling with helpful error messages
- **Unit Tests**: Full test coverage for all modules
- **Modern Web UI**: Drag-and-drop React frontend for batch video upload, options, and progress tracking (optional)

## Installation

### Option 1: Manual Installation

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

### Option 2: Docker (Recommended)

For easy deployment and consistent environments, use Docker:

1. **Install Docker Desktop** (if not already installed)
2. **Clone the repository:**
   ```bash
   git clone https://github.com/zuche-ai/ai-video-translator.git
   cd ai-video-translator
   ```

3. **Start with Docker:**
   ```bash
   # Using the helper script
   ./docker-run.sh
   
   # Or manually
   docker-compose up --build -d
   ```

4. **Access the application:**
   - Frontend UI: http://localhost:3000
   - Backend API: http://localhost:5001

See [DOCKER.md](DOCKER.md) for detailed Docker documentation.

## Usage

You can use **either the command-line interface (CLI)** (see below), **or the local Web UI** (see next section) for video translation and voice cloning.

### Web UI (React Frontend)

The Web UI provides a modern, local interface for batch video upload, drag-and-drop, and real-time progress tracking. All processing happens locally—no data leaves your machine.

#### 1. Install frontend dependencies

```bash
cd frontend
npm install
```

#### 2. Start the frontend (React/Vite)

```bash
npm run dev
```

- The UI will be available at [http://localhost:5173](http://localhost:5173)
- Make sure the backend (Flask API) is running (see below)

#### 3. Start the backend (Flask API)

In a separate terminal:

```bash
python -m video_translator.api
```

- The backend will run at [http://localhost:5001](http://localhost:5001)

#### 4. Using the UI

- Drag and drop video files or use the file picker
- Select source/target language, audio mode, and options
- Click **Start Processing**
- Progress bars and download links will appear for each video
- Supports batch processing, voice cloning, overlay, and captions

#### Features of the Web UI
- Drag-and-drop or batch file selection
- All CLI options available as dropdowns/toggles
- Real-time progress bars for each video
- Download links for processed videos
- Local-only: No uploads to any server

---

### Command-Line Usage (CLI)

### Basic Usage (Subtitles Only)

```bash
python -m video_translator.main --input video.mp4 --src-lang en --tgt-lang es --output translated_video.mp4
```

### Voice Cloning Usage

```bash
# Replace original audio with cloned voice
python -m video_translator.main --input video.mp4 --src-lang en --tgt-lang es --output translated_video.mp4 --voice-clone --audio-mode replace

# Overlay translated audio on original audio
python -m video_translator.main --input video.mp4 --src-lang en --tgt-lang es --output translated_video.mp4 --voice-clone --audio-mode overlay --original-volume 0.3

# Subtitles only (default)
python -m video_translator.main --input video.mp4 --src-lang en --tgt-lang es --output translated_video.mp4 --audio-mode subtitles-only
```

### New Features

- `--add-captions`: Burn subtitles even when using overlay or replace audio modes (captions + voice overlay together)
- `--caption-font-size`: Set the font size for subtitles (default: 24, smaller and less intrusive)

### Usage Example (Overlay + Captions, Small Font)

```bash
python -m video_translator.main --input video.mp4 --src-lang en --tgt-lang es --output translated_video.mp4 \
  --voice-clone --audio-mode overlay --add-captions --caption-font-size 24
```

### Parameters

- `--input`: Path to the input video file
- `--src-lang`: Source language code (e.g., `en`, `fr`, `de`)
- `--tgt-lang`: Target language code for subtitles and voice cloning
- `--output`: Path for the output video file
- `--voice-clone`: Enable voice cloning with translated audio
- `--audio-mode`: Audio processing mode (`replace`, `overlay`, `subtitles-only`)
- `--original-volume`: Volume of original audio when overlaying (0.0 to 1.0, default: 0.3)
- `--reference-audio`: Path to reference audio file for voice cloning (optional)
- `--debug`: Enable debug output (optional)
- `--add-captions`: Burn subtitles even when using overlay or replace audio modes
- `--caption-font-size`: Font size for subtitles (default: 24)

### Audio Modes

1. **`subtitles-only`** (default): Only add translated subtitles, keep original audio
2. **`replace`**: Replace original audio with cloned voice speaking translated text
3. **`overlay`**: Overlay translated audio on original audio (original at reduced volume)

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
# Basic subtitle translation
python -m video_translator.main --input my_video.mp4 --src-lang en --tgt-lang es --output spanish_video.mp4

# Voice cloning with audio replacement
python -m video_translator.main --input my_video.mp4 --src-lang en --tgt-lang es --output spanish_video.mp4 --voice-clone --audio-mode replace

# Voice cloning with audio overlay (original audio at 20% volume)
python -m video_translator.main --input my_video.mp4 --src-lang en --tgt-lang es --output spanish_video.mp4 --voice-clone --audio-mode overlay --original-volume 0.2

# French to German with voice cloning and debug output
python -m video_translator.main --input french_video.mp4 --src-lang fr --tgt-lang de --output german_video.mp4 --voice-clone --audio-mode replace --debug
```

## Voice Cloning Technology

The voice cloning feature uses **Coqui TTS** with the **XTTS v2** model, which provides:

- **Local Processing**: All voice cloning happens locally - no API calls or data sent to external services
- **High Quality**: State-of-the-art voice cloning with natural-sounding results
- **Multilingual Support**: Supports multiple languages for voice cloning
- **Voice Profiling**: Automatically extracts voice characteristics from the original audio
- **Batch Processing**: Efficiently processes multiple text segments with proper timing

### How Voice Cloning Works

1. **Voice Extraction**: Extracts audio from the original video
2. **Voice Profiling**: Analyzes the first 10 seconds (configurable) to create a voice profile
3. **Text Translation**: Translates the transcribed text to the target language
4. **Voice Cloning**: Generates translated audio using the cloned voice
5. **Audio Synchronization**: Aligns translated audio with original timing
6. **Video Processing**: Combines video, subtitles, and translated audio

## Error Handling

The application provides comprehensive error handling:

- **File Not Found**: Clear messages when input files don't exist
- **Missing Dependencies**: Helpful installation instructions for missing packages
- **Invalid Input**: Validation of input parameters and data
- **Processing Failures**: Detailed error messages for processing issues
- **Voice Cloning Errors**: Specific error handling for TTS and audio processing
- **Debug Mode**: Additional debugging information when `--debug` is used

### Common Error Solutions

1. **"ffmpeg not found"**: Install FFmpeg using your system's package manager
2. **"Whisper is not installed"**: Run `pip install openai-whisper`
3. **"TTS model not found"**: The Coqui TTS model will be downloaded automatically on first use
4. **"Video file not found"**: Check the file path and ensure the video exists
5. **"No segments provided"**: The video may not contain speech or the audio is too quiet
6. **"Voice cloning failed"**: Ensure sufficient audio quality and length for voice profiling

## Testing

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run individual test modules
python -m unittest tests.test_transcriber
python -m unittest tests.test_translator
python -m unittest tests.test_subtitles
python -m unittest tests.test_video_editor
python -m unittest tests.test_voice_cloner
python -m unittest tests.test_main_audio_sync
python -m unittest tests.test_audio_sync
```

### Test Coverage

The test suite covers:

- **Core Modules**: Audio transcription, translation, and subtitle generation (`video_translator/core/`)
- **Audio Modules**: Voice cloning and audio processing (`video_translator/audio/`)
- **Video Modules**: Subtitle burning and video processing (`video_translator/video/`)
- **Main Module**: Command-line interface and error handling
- **Audio Synchronization**: Timeline and overlay logic for translated audio

Each module includes tests for:
- ✅ Successful operations
- ❌ Error conditions
- 🔍 Input validation
- 🐛 Debug mode functionality

## Project Structure

```
video_translator/
├── video_translator/
│   ├── __init__.py
│   ├── main.py                # Main application entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── transcriber.py     # Audio transcription using Whisper
│   │   ├── translator.py     # Text translation using ArgosTranslate
│   │   └── subtitles.py      # SRT subtitle file generation
│   ├── audio/
│   │   ├── __init__.py
│   │   └── voice_cloner.py   # Voice cloning using Coqui TTS
│   └── video/
│       ├── __init__.py
│       └── video_editor.py   # Video processing with FFmpeg
├── tests/
│   ├── __init__.py
│   ├── test_transcriber.py
│   ├── test_translator.py
│   ├── test_subtitles.py
│   ├── test_video_editor.py
│   ├── test_voice_cloner.py
│   ├── test_main.py
│   ├── test_main_audio_sync.py
│   └── test_audio_sync.py
├── requirements.txt           # Python dependencies
├── run_tests.py               # Test runner
├── .gitignore                 # Git ignore rules
├── SECURITY.md                # Security best practices
└── README.md                  # This file
```

## Dependencies

- **openai-whisper**: Audio transcription
- **argostranslate**: Text translation
- **pysrt**: Subtitle file handling
- **ffmpeg-python**: Video processing
- **tqdm**: Progress bars
- **TTS**: Voice cloning (Coqui TTS)
- **torch**: PyTorch for TTS models
- **torchaudio**: Audio processing for PyTorch
- **numpy**: Numerical computing
- **librosa**: Audio analysis
- **soundfile**: Audio file I/O
- **webrtcvad**: Voice activity detection (for reference audio quality)

## Performance Considerations

- **Voice Cloning**: The first run will download the TTS model (~2GB), subsequent runs will be faster
- **Processing Time**: Voice cloning adds significant processing time, especially for longer videos
- **Memory Usage**: TTS models require substantial RAM (4GB+ recommended)
- **GPU Acceleration**: Voice cloning can use GPU acceleration if available (CUDA)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `python run_tests.py`
6. Submit a pull request

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

### Sync Guarantee

- **Captions and voice overlay are always in sync**: Both use the same timestamped segments from the transcription and translation pipeline, so the displayed text matches the spoken audio.
