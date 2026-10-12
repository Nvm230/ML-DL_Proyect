import os
import sys
import time
import urllib.request
import subprocess

URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
TARGET_DIR = "/home/nvm/Documents/UTEC/MLDP/data/food101"
TAR_PATH = os.path.join(TARGET_DIR, "food-101.tar.gz")
EXTRACT_DIR = os.path.join(TARGET_DIR, "food-101")

os.makedirs(TARGET_DIR, exist_ok=True)

if os.path.exists(EXTRACT_DIR) and os.path.exists(os.path.join(EXTRACT_DIR, "images")):
    print("Food-101 already extracted and ready!")
    sys.exit(0)

expected_size = 4996278331

def extract_archive():
    print("Extracting archive with system tar (fast)...")
    t0 = time.time()
    subprocess.run(["tar", "-xzf", TAR_PATH, "-C", TARGET_DIR], check=True)
    print(f"Extraction completed successfully in {time.time()-t0:.1f}s!")

if os.path.exists(TAR_PATH):
    current_size = os.path.getsize(TAR_PATH)
    if current_size == expected_size:
        print("Tarfile already downloaded completely. Extracting...")
        extract_archive()
        sys.exit(0)

print(f"Downloading {URL} to {TAR_PATH} (resuming if partial)...")
headers = {'User-Agent': 'Mozilla/5.0'}

resume_byte_pos = os.path.getsize(TAR_PATH) if os.path.exists(TAR_PATH) else 0
req = urllib.request.Request(URL, headers=headers)
if resume_byte_pos > 0:
    req.add_header("Range", f"bytes={resume_byte_pos}-")
    mode = 'ab'
else:
    mode = 'wb'

try:
    with urllib.request.urlopen(req) as response, open(TAR_PATH, mode) as out_file:
        downloaded = resume_byte_pos
        start_time = time.time()
        last_log = start_time
        
        while True:
            buffer = response.read(1024 * 1024)  # 1MB chunks
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            
            now = time.time()
            if now - last_log >= 10:
                speed = (downloaded - resume_byte_pos) / (now - start_time) / (1024 * 1024)
                pct = (downloaded / expected_size) * 100
                print(f"[{time.strftime('%X')}] Downloaded {downloaded / (1024**3):.2f} / {expected_size / (1024**3):.2f} GB ({pct:.1f}%) - Speed: {speed:.2f} MB/s", flush=True)
                last_log = now

    print("Download completed! Now extracting...")
    extract_archive()
except Exception as e:
    print(f"Error during download/extract: {e}", file=sys.stderr)
    sys.exit(1)
