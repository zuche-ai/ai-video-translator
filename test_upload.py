#!/usr/bin/env python3
import requests
import time
import os

def test_video_upload():
    # Test video file path - you can replace this with any video file
    video_file = "test_video.mp4"
    
    # Check if test video exists, if not create a simple one
    if not os.path.exists(video_file):
        print("Creating test video file...")
        os.system("ffmpeg -f lavfi -i testsrc=duration=5:size=320x240:rate=1 -f lavfi -i sine=frequency=1000:duration=5 -c:v libx264 -c:a aac test_video.mp4")
    
    # Upload the video
    url = "http://localhost:3000/api/process"
    
    files = {
        'video': ('test_video.mp4', open(video_file, 'rb'), 'video/mp4')
    }
    
    data = {
        'src_lang': 'en',
        'tgt_lang': 'es',
        'voice_clone': 'true',
        'audio_mode': 'overlay',
        'original_volume': '0.3',
        'add_captions': 'true',
        'caption_font_size': '24'
    }
    
    print("Uploading video...")
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        job_id = response.json()['job_id']
        print(f"Job started with ID: {job_id}")
        
        # Monitor progress
        status_url = f"http://localhost:3000/api/status/{job_id}"
        
        while True:
            status_response = requests.get(status_url)
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Status: {status_data['status']}, Progress: {status_data['progress']}%")
                
                if status_data['status'] == 'done':
                    print("✅ Processing completed successfully!")
                    # Download the result
                    result_url = f"http://localhost:3000/api/result/{job_id}"
                    result_response = requests.get(result_url)
                    if result_response.status_code == 200:
                        with open(f"output_{job_id}.mp4", "wb") as f:
                            f.write(result_response.content)
                        print(f"✅ Result saved as output_{job_id}.mp4")
                    break
                elif status_data['status'] == 'error':
                    print(f"❌ Processing failed: {status_data.get('error', 'Unknown error')}")
                    break
            else:
                print(f"Failed to get status: {status_response.status_code}")
                break
            
            time.sleep(3)  # Check every 3 seconds
    else:
        print(f"Failed to upload: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_video_upload() 