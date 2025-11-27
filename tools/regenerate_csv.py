#!/usr/bin/env python3
"""
Master Dataset Generator
1. Scans for audio.
2. (Optional) Renames files to hash for deduplication.
3. Reads audio bytes.
4. Splits into chunks using HUMAN-READABLE 1-based indexing.
"""

import pandas as pd
from pathlib import Path
import argparse
import shutil
from datetime import datetime
import hashlib
import math

def compute_file_hash(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:8]

def get_file_size_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)

def regenerate_metadata(base_dir: str, global_rename: bool = None):
    base_path = Path(base_dir)
    metadata_path = base_path / "metadata.csv"
    data_dir = base_path / "data"
    audio_dir = base_path / "audio"
    transcripts_dir = base_path / "transcripts"
    MISSING_TRANSCRIPT_PLACEHOLDER = "!!! MISSING TRANSCRIPT FILE !!!"
    
    # Hugging Face recommended chunk size (500MB)
    TARGET_CHUNK_SIZE_MB = 500 

    print("=" * 70)
    print(f"PROCESSING: {base_path}")
    print("=" * 70)

    if not audio_dir.exists():
        print(f"\n✗ ERROR: audio directory not found at: {audio_dir}")
        return False

    # 1. Scan Files
    print(f"\n🔍 Scanning audio files...")
    audio_files = sorted(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")))
    if not audio_files:
        print(f"\n✗ ERROR: No audio files found.")
        return False
    
    total_size_mb = sum(get_file_size_mb(f) for f in audio_files)
    print(f"   ✓ Found {len(audio_files)} files (Approx {total_size_mb:.2f} MB total)")

    # 2. Renaming Logic
    should_rename = global_rename
    if should_rename is None:
        print("\n" + "-" * 30)
        print("RENAME FILES (Deduplication)?")
        print("-" * 30)
        rename_response = input("Rename files to hash? (yes/no): ").strip().lower()
        should_rename = rename_response in ['yes', 'y']

    rename_mapping = {}
    duplicates_to_delete = []

    if should_rename:
        print(f"\n🔨 Computing hashes...")
        hash_to_file = {}
        for f_wav in audio_files:
            file_hash = compute_file_hash(f_wav)
            if file_hash in hash_to_file:
                duplicates_to_delete.append(f_wav)
            else:
                hash_to_file[file_hash] = f_wav
                rename_mapping[f_wav.name] = f"{file_hash}{f_wav.suffix}"

    # 3. Build Dataset in Memory
    new_data = []
    files_to_process = [f for f in audio_files if f not in duplicates_to_delete]
    
    print(f"\n📚 Reading audio & transcripts...")
    
    for f_wav in files_to_process:
        old_file_name = f_wav.name
        new_file_name = rename_mapping.get(old_file_name, old_file_name) if should_rename else old_file_name

        # Read Transcript
        possible_txt = [Path(old_file_name).stem + ".txt", Path(new_file_name).stem + ".txt"]
        transcription = MISSING_TRANSCRIPT_PLACEHOLDER
        for txt_name in possible_txt:
            txt_path = transcripts_dir / txt_name
            if txt_path.exists():
                try:
                    transcription = txt_path.read_text(encoding='utf-8').strip()
                    break 
                except: continue

        # Read Audio Bytes
        try:
            with open(f_wav, "rb") as audio_f:
                audio_bytes = audio_f.read()
        except Exception as e:
            print(f"   ⚠️ Skipping broken file {f_wav.name}: {e}")
            continue

        new_data.append({
            'file_name': new_file_name, 
            'transcription': transcription,
            'audio': {'bytes': audio_bytes, 'path': new_file_name}
        })

    # 4. Execute Renaming/Deletion
    print(f"\n⚠️  Processed {len(new_data)} valid entries.")
    if input(f"Proceed with saving? (yes/no): ").strip().lower() not in ['yes', 'y']:
        return False

    if should_rename and duplicates_to_delete:
        for f in duplicates_to_delete: 
            f.unlink()
            (transcripts_dir / (f.stem+".txt")).unlink(missing_ok=True)
            
    if should_rename:
        print("   Renaming files on disk...")
        for old, new in rename_mapping.items():
            if old == new: continue
            (audio_dir/old).rename(audio_dir/new)
            old_txt = transcripts_dir / (Path(old).stem + ".txt")
            if old_txt.exists(): old_txt.rename(transcripts_dir / (Path(new).stem + ".txt"))
        
        pd.DataFrame(list(rename_mapping.items()), columns=['old','new']).to_csv(base_path/"filename_mapping.csv", index=False)

    # 5. Save Metadata CSV
    df_full = pd.DataFrame(new_data)
    df_full[['file_name', 'transcription']].to_csv(metadata_path, index=False)
    print(f"\n✅ Created metadata.csv")

    # 6. Save SHARDED Parquet Files (1-BASED INDEXING)
    if data_dir.exists(): shutil.rmtree(data_dir)
    data_dir.mkdir()

    avg_row_size_mb = total_size_mb / len(new_data)
    rows_per_chunk = int(TARGET_CHUNK_SIZE_MB / avg_row_size_mb)
    rows_per_chunk = max(1, rows_per_chunk)
    
    num_chunks = math.ceil(len(new_data) / rows_per_chunk)
    print(f"\n📦 Splitting into {num_chunks} files (~{TARGET_CHUNK_SIZE_MB}MB each)...")

    # Start loop at 1 instead of 0
    for i in range(1, num_chunks + 1):
        start_idx = (i - 1) * rows_per_chunk
        end_idx = start_idx + rows_per_chunk
        chunk_df = df_full.iloc[start_idx:end_idx]
        
        # Format: train-00001-of-00013.parquet
        output_name = f"train-{str(i).zfill(5)}-of-{str(num_chunks).zfill(5)}.parquet"
        chunk_path = data_dir / output_name
        
        chunk_df.to_parquet(chunk_path, index=False)
        print(f"   ✓ Saved {output_name} ({len(chunk_df)} rows)")

    print(f"\n✅ SUCCESS! Dataset saved in folder: /data")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', required=True)
    args = parser.parse_args()
    regenerate_metadata(args.base_dir)