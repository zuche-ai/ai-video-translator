# Video Translator

A self-contained Python CLI tool to:
- Transcribe any video in a specified language (using Whisper)
- Translate the transcription to a target language (using Argos Translate)
- Generate and burn translated captions into the original video (using ffmpeg)

## Features
- Fully offline, open-source workflow
- Supports any language Whisper and Argos Translate support
- Optional `--debug` flag for command transparency
- Progress feedback at each step

## Usage

1. Place your video file in this directory (or specify a path).
2. Run:

```sh
python main.py --input myvideo.mp4 --src-lang en --tgt-lang es --output myvideo_es.mp4 [--debug]
```

- `--input`: Path to the input video file
- `--src-lang`: Source language code (e.g., en, fr, de)
- `--tgt-lang`: Target language code for captions
- `--output`: Path for the output video file
- `--debug`: (Optional) Print all commands and actions

## Requirements
- Python 3.8+
- ffmpeg (install via Homebrew: `brew install ffmpeg`)

## Installation
Install Python dependencies:
```sh
pip install -r requirements.txt
```

Download Argos Translate language models as needed (see their docs).
