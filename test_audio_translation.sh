#!/bin/bash
set -e

# CONFIG
AUDIO_FILE="christtestshort.wav"
REFERENCE_AUDIO="christtestshort.wav"  # Using same file for reference; change if you have a different one
SRC_LANG="en"
TGT_LANG="es"
API_URL="http://localhost:5001/translate_audio"
STATUS_URL="http://localhost:5001/status"
RESULT_URL="http://localhost:5001/result"

# 1. Build and start Docker containers

echo "[INFO] Building and starting Docker containers..."
docker compose -f docker/docker-compose.yml up -d --build

# 2. Wait for backend to be healthy

echo "[INFO] Waiting for backend to be healthy..."
for i in {1..30}; do
    STATUS=$(curl -s http://localhost:5001/health | grep '"status"' || true)
    if [[ "$STATUS" == *"healthy"* ]]; then
        echo "[INFO] Backend is healthy."
        break
    fi
    sleep 2
done

# 3. Send audio file to /translate_audio endpoint

echo "[INFO] Sending $AUDIO_FILE to /translate_audio..."
RESPONSE=$(curl -s -F "audio=@$AUDIO_FILE" -F "reference_audio=@$REFERENCE_AUDIO" \
    -F "src_lang=$SRC_LANG" -F "tgt_lang=$TGT_LANG" \
    "$API_URL")

JOB_ID=$(echo "$RESPONSE" | grep -o '"job_id"[^"]*"[^"]*"' | cut -d '"' -f4)
if [ -z "$JOB_ID" ]; then
    echo "[ERROR] Failed to get job_id from response: $RESPONSE"
    exit 1
fi

echo "[INFO] Job ID: $JOB_ID"

# 4. Poll for job completion

echo "[INFO] Polling for job completion..."
for i in {1..60}; do
    STATUS_JSON=$(curl -s "$STATUS_URL/$JOB_ID")
    STATUS=$(echo "$STATUS_JSON" | grep -o '"status"[^"]*"[^"]*"' | cut -d '"' -f4)
    echo "[INFO] Status: $STATUS"
    if [ "$STATUS" == "done" ]; then
        break
    elif [ "$STATUS" == "error" ]; then
        echo "[ERROR] Job failed: $STATUS_JSON"
        exit 1
    fi
    sleep 5
done

# 5. Download the result

echo "[INFO] Downloading result..."
RESULT_FILE="translated_${AUDIO_FILE%.wav}_${SRC_LANG}_${TGT_LANG}.wav"

# Try to download from API first
if curl -s -o "$RESULT_FILE" "$RESULT_URL/$JOB_ID"; then
    echo "[SUCCESS] Translated audio downloaded via API as $RESULT_FILE"
else
    echo "[WARNING] API download failed, trying to copy from container..."
    # Fallback: copy directly from container
    if docker cp "video-translator-backend:/tmp/${AUDIO_FILE%.wav}_${SRC_LANG}_${TGT_LANG}_translated.wav" "$RESULT_FILE"; then
        echo "[SUCCESS] Translated audio copied from container as $RESULT_FILE"
    else
        echo "[ERROR] Failed to get translated audio from container."
        echo "[INFO] Checking available files in container:"
        docker exec video-translator-backend ls -la /tmp/ | grep -E "(translated|${AUDIO_FILE%.wav})" || true
        exit 1
    fi
fi

# Verify the file has content
if [ -f "$RESULT_FILE" ] && [ -s "$RESULT_FILE" ]; then
    DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$RESULT_FILE" 2>/dev/null || echo "unknown")
    echo "[SUCCESS] Translated audio saved as $RESULT_FILE (duration: ${DURATION}s)"
else
    echo "[ERROR] Downloaded file is empty or missing."
    exit 1
fi 