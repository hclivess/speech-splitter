#!/usr/bin/env python3
"""
Regenerate metadata.csv from scratch by scanning for all existing audio files 
and automatically retrieving transcriptions from corresponding .txt files.

Audio file: base_dir/speaker_name/audio/file_name.wav
Transcription file: base_dir/speaker_name/transcripts/file_name.txt

Optional: Rename files to hash-based names (first 8 chars of SHA256).
"""

import pandas as pd
from pathlib import Path
import argparse
import shutil
from datetime import datetime
import hashlib


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file and return first 8 characters."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:8]


def regenerate_metadata_csv(base_dir: str, speaker_name: str):
    """
    Regenerate metadata.csv by scanning for audio files and reading transcripts.
    
    Args:
        base_dir: Base output directory
        speaker_name: Speaker folder name
    """
    
    base_path = Path(base_dir) / speaker_name
    metadata_path = base_path / "metadata.csv"
    audio_dir = base_path / "audio"
    transcripts_dir = base_path / "transcripts"
    
    # Placeholder for missing transcripts
    MISSING_TRANSCRIPT_PLACEHOLDER = "!!! MISSING TRANSCRIPT FILE !!!"
    
    print("="*70)
    print("METADATA.CSV AUTOMATED REGENERATION TOOL")
    print("="*70)
    print(f"\nSpeaker: {speaker_name}")
    print(f"Metadata (will be created/overwritten): {metadata_path}")
    print(f"Audio dir: {audio_dir}")
    print(f"Transcripts dir: {transcripts_dir}")
    
    # Check if directories exist
    if not audio_dir.exists():
        print(f"\n❌ ERROR: audio directory not found at: {audio_dir}")
        return False
    
    if not transcripts_dir.exists():
        print(f"\n❌ WARNING: transcripts directory not found at: {transcripts_dir}")
        print(f"           Using '{MISSING_TRANSCRIPT_PLACEHOLDER}' for all entries.")
        
    # Find all .wav files
    print(f"\n🔍 Scanning for audio files in {audio_dir}...")
    audio_files = sorted(list(audio_dir.glob("*.wav")))
    
    if not audio_files:
        print(f"\n❌ ERROR: No .wav files found in the audio directory.")
        return False
        
    print(f"   ✓ Found {len(audio_files)} .wav files.")

    # Ask about renaming
    print("\n" + "="*70)
    print("FILE RENAMING OPTION")
    print("="*70)
    print("Would you like to rename all files to hash-based names?")
    print("  - Files will be renamed using first 8 chars of SHA256 hash")
    print("  - Both .wav and .txt files will be renamed to match")
    print("  - A mapping file will be created: filename_mapping.csv")
    print("  - Example: old_recording_123.wav → a3f9b2c1.wav")
    
    rename_response = input("\nRename files? (yes/no): ").strip().lower()
    should_rename = rename_response in ['yes', 'y']
    
    rename_mapping = {}
    
    if should_rename:
        print(f"\n🔄 Computing hashes and preparing to rename {len(audio_files)} files...")
        
        # Check for hash collisions first
        hash_to_files = {}
        for f_wav in audio_files:
            file_hash = compute_file_hash(f_wav)
            if file_hash in hash_to_files:
                print(f"\n❌ ERROR: Hash collision detected!")
                print(f"   Files: {hash_to_files[file_hash]} and {f_wav.name}")
                print(f"   Both have hash: {file_hash}")
                print(f"   This is extremely rare. Please report this issue.")
                return False
            hash_to_files[file_hash] = f_wav.name
            rename_mapping[f_wav.name] = f"{file_hash}.wav"
        
        print(f"   ✓ No hash collisions detected.")
        print(f"\nPreview of renaming (first 5):")
        for i, (old_name, new_name) in enumerate(list(rename_mapping.items())[:5]):
            print(f"   {old_name} → {new_name}")
        if len(rename_mapping) > 5:
            print(f"   ... and {len(rename_mapping) - 5} more")
        
        confirm_rename = input(f"\n⚠️ Proceed with renaming? (yes/no): ").strip().lower()
        if confirm_rename not in ['yes', 'y']:
            print("\n❌ Renaming cancelled. Proceeding without renaming.")
            should_rename = False
            rename_mapping = {}

    # Prepare data for new CSV
    new_data = []
    missing_count = 0
    
    print(f"\n📚 Processing files...")
    
    for f_wav in audio_files:
        old_file_name = f_wav.name
        
        # Determine new filename
        if should_rename:
            new_file_name = rename_mapping[old_file_name]
            new_wav_path = audio_dir / new_file_name
        else:
            new_file_name = old_file_name
            new_wav_path = f_wav
        
        # Find transcript file (using old name for lookup)
        transcript_name = Path(old_file_name).stem + ".txt"
        transcript_path = transcripts_dir / transcript_name
        
        transcription = MISSING_TRANSCRIPT_PLACEHOLDER
        
        if transcript_path.exists():
            try:
                transcription = transcript_path.read_text(encoding='utf-8').strip()
            except Exception as e:
                print(f"   ⚠️ Could not read {transcript_name}: {e}")
                missing_count += 1
        else:
            missing_count += 1

        new_data.append([new_file_name, transcription])

    # Report missing files
    if missing_count > 0:
        print(f"   ❌ WARNING: {missing_count} corresponding .txt files were not found or could not be read.")
        print(f"              These entries will use the placeholder: '{MISSING_TRANSCRIPT_PLACEHOLDER}'")
    else:
        print("   ✓ All transcriptions found and loaded.")


    # Confirm regeneration
    print(f"\n⚠️  This will:")
    print(f"   1. Create a new metadata.csv with {len(new_data)} entries.")
    print(f"   2. Use content from corresponding .txt files.")
    if should_rename:
        print(f"   3. Rename {len(rename_mapping)} audio files and their transcripts.")
        print(f"   4. Create filename_mapping.csv with old → new name mappings.")
    if metadata_path.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = base_path / f"metadata.csv.backup.regen.{timestamp}"
        backup_num = 5 if should_rename else 3
        print(f"   {backup_num}. Create backup of existing CSV: {backup_path.name}")
    
    response = input(f"\n⚠️ Proceed with {'renaming and ' if should_rename else ''}regeneration? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\n❌ Operation cancelled.")
        return False
    
    # Perform file renaming if requested
    if should_rename:
        print(f"\n🔄 Renaming files...")
        renamed_count = 0
        
        for old_name, new_name in rename_mapping.items():
            old_wav = audio_dir / old_name
            new_wav = audio_dir / new_name
            
            old_txt = transcripts_dir / (Path(old_name).stem + ".txt")
            new_txt = transcripts_dir / (Path(new_name).stem + ".txt")
            
            try:
                # Rename audio file
                old_wav.rename(new_wav)
                
                # Rename transcript file if it exists
                if old_txt.exists():
                    old_txt.rename(new_txt)
                
                renamed_count += 1
            except Exception as e:
                print(f"   ❌ ERROR renaming {old_name}: {e}")
                return False
        
        print(f"   ✓ Renamed {renamed_count} audio files (and matching transcripts).")
        
        # Save mapping file
        mapping_path = base_path / "filename_mapping.csv"
        mapping_df = pd.DataFrame(list(rename_mapping.items()), columns=['old_filename', 'new_filename'])
        mapping_df.to_csv(mapping_path, index=False)
        print(f"   ✓ Mapping saved to: {mapping_path.name}")
    
    # Create backup if file exists
    if metadata_path.exists():
        try:
            shutil.copy2(metadata_path, backup_path)
            print(f"\n💾 Backup created: {backup_path.name}")
        except Exception as e:
            print(f"\n❌ ERROR: Could not create backup: {e}")
            return False
    
    # Create dataframe
    df_new = pd.DataFrame(new_data, columns=['file_name', 'transcription'])
    
    # Save new CSV
    df_new.to_csv(metadata_path, index=False)
    print(f"\n✅ New metadata.csv saved!")
    print(f"   Generated {len(df_new)} entries.")
    
    # Summary
    print(f"\n{'='*70}")
    print("AUTOMATED REGENERATION COMPLETE")
    print("="*70)
    print(f"Files scanned: {len(audio_files)}")
    print(f"Entries generated: {len(df_new)}")
    if should_rename:
        print(f"Files renamed: {len(rename_mapping)}")
        print(f"Mapping file: filename_mapping.csv")
    if missing_count > 0:
        print(f"*** WARNING: {missing_count} ENTRIES NEED MANUAL REVIEW/CORRECTION ***")
    print("="*70 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate metadata.csv from scratch by scanning for all existing audio files and reading corresponding .txt transcripts"
    )
    parser.add_argument(
        '--base-dir',
        required=True,
        help='Base output directory (e.g., C:/ChatterboxTraining/output)'
    )
    parser.add_argument(
        '--speaker',
        required=True,
        help='Speaker folder name (e.g., "00 Prolog")'
    )
    
    args = parser.parse_args()
    
    success = regenerate_metadata_csv(
        base_dir=args.base_dir,
        speaker_name=args.speaker
    )
    
    if success:
        print("✓ CSV regenerated. Check for '!!! MISSING TRANSCRIPT FILE !!!' placeholders.")
    else:
        print("✗ Regeneration failed")


if __name__ == "__main__":
    main()
