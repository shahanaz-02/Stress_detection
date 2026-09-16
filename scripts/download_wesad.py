"""
Robust Resumable Downloader for Authentic WESAD Dataset
Official Link: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
Supports HTTP Range resumes and chunked downloading with automatic retry logic.
"""

import os
import sys
import ssl
import time
import zipfile
import urllib.request

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
WESAD_DIR = os.path.join(RAW_DIR, "WESAD")
ZIP_PATH = os.path.join(RAW_DIR, "WESAD.zip")
DOWNLOAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"

WESAD_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

def download_resumable(url, destination_path):
    ctx = ssl._create_unverified_context()
    
    # Get total file size
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            total_bytes = int(response.headers.get('Content-Length', 0))
    except Exception as e:
        print(f"[ERROR] Could not fetch headers: {e}")
        total_bytes = 0

    downloaded_bytes = os.path.getsize(destination_path) if os.path.exists(destination_path) else 0

    if total_bytes > 0 and downloaded_bytes >= total_bytes:
        print(f"[INFO] File already fully downloaded ({downloaded_bytes / (1024*1024):.1f} MB).")
        return True

    chunk_size = 1024 * 1024  # 1 MB chunk
    max_retries = 100
    retry_count = 0

    print(f"[INFO] Total Size: {total_bytes / (1024*1024):.1f} MB | Resume Offset: {downloaded_bytes / (1024*1024):.1f} MB")

    while downloaded_bytes < total_bytes or total_bytes == 0:
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Range': f'bytes={downloaded_bytes}-'
            })
            
            with urllib.request.urlopen(req, context=ctx, timeout=30) as response, open(destination_path, 'ab') as f:
                if total_bytes == 0:
                    total_bytes = int(response.headers.get('Content-Range', '').split('/')[-1] or 0)
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    f.flush()
                    downloaded_bytes += len(chunk)
                    
                    if total_bytes > 0:
                        pct = (downloaded_bytes / total_bytes) * 100
                        mb_dn = downloaded_bytes / (1024 * 1024)
                        mb_tot = total_bytes / (1024 * 1024)
                        sys.stdout.write(f"\r[DOWNLOADING WESAD] {pct:.2f}% ({mb_dn:.1f} MB / {mb_tot:.1f} MB)")
                        sys.stdout.flush()
                        
            if downloaded_bytes >= total_bytes and total_bytes > 0:
                print(f"\n[SUCCESS] Download completed ({downloaded_bytes / (1024*1024):.1f} MB).")
                return True

        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"\n[ERROR] Download failed after {max_retries} retries: {e}")
                return False
            time.sleep(2)
            sys.stdout.write(f"\n[WARNING] Connection interrupted: {e}. Retrying ({retry_count}/{max_retries}) from {downloaded_bytes / (1024*1024):.1f} MB...\n")
            sys.stdout.flush()

    return downloaded_bytes >= total_bytes

def extract_and_verify():
    print(f"[INFO] Extracting {ZIP_PATH} to {RAW_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)
    print(f"[SUCCESS] Extracted WESAD dataset archive.")
    
    # Verify subjects
    found = [s for s in WESAD_SUBJECTS if os.path.exists(os.path.join(WESAD_DIR, s, f"{s}.pkl")) or os.path.exists(os.path.join(RAW_DIR, "WESAD", s, f"{s}.pkl"))]
    print(f"[VERIFY] Found {len(found)}/{len(WESAD_SUBJECTS)} subject pickle files.")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(WESAD_DIR, exist_ok=True)
    
    # Check if subjects already extracted
    existing_pkls = [s for s in WESAD_SUBJECTS if os.path.exists(os.path.join(WESAD_DIR, s, f"{s}.pkl"))]
    if len(existing_pkls) == len(WESAD_SUBJECTS):
        print(f"[INFO] All {len(WESAD_SUBJECTS)} raw WESAD subject pickle files are present.")
        return
        
    success = download_resumable(DOWNLOAD_URL, ZIP_PATH)
    if success:
        extract_and_verify()

if __name__ == "__main__":
    main()
