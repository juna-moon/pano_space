#!/usr/bin/env python3
import os
import zipfile
import argparse

def extract_and_cleanup(directory):
    """
    Extracts all .zip files in the given directory into subfolders
    named after each archive (without the .zip extension), then
    deletes the original .zip file upon successful extraction.
    """
    for fname in os.listdir(directory):
        if not fname.lower().endswith('.zip'):
            continue
        zip_path = os.path.join(directory, fname)
        target_dir = os.path.join(directory, fname[:-4])
        print(f"Extracting {zip_path} → {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(target_dir)
            os.remove(zip_path)
            print(f"✔ Deleted archive {zip_path}")
        except Exception as e:
            print(f"✖ Failed to process {zip_path}: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract all zip files in a directory and delete them after extraction."
    )
    ap.add_argument("directory", help="Path to folder containing .zip files")
    args = ap.parse_args()
    extract_and_cleanup(args.directory)