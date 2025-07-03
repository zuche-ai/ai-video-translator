#!/bin/bash

# Directory containing the MP3 files
INPUT_DIR="/Users/iaingreer/Documents/TrumanMadsenLecturesJosephSmith"
# Output directory for translated files (optional, can be same as input)
OUTPUT_DIR="$INPUT_DIR/translated"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all mp3 files in the input directory
for input_file in "$INPUT_DIR"/*.mp3; do
  # Get the base filename without extension
  base_name=$(basename "$input_file" .mp3)
  # Set the output file path
  output_file="$OUTPUT_DIR/${base_name}_es.mp3"
  echo "Translating: $input_file -> $output_file"
  # Run the translation command
  python -m video_translator.audio_translation.cli \
    --input "$input_file" \
    --output "$output_file" \
    --src-lang en \
    --tgt-lang es \
    --voice-clone
  echo "Done: $output_file"
done

echo "Batch translation complete! Output files are in $OUTPUT_DIR" 