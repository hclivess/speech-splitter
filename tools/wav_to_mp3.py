#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import argparse

# ==================== CONFIGURATION ====================
# You can change these defaults or use command-line arguments
DEFAULT_QUALITY = "2"  # MP3 quality: 0-9 (lower is better quality)
# =======================================================

def check_ffmpeg():
    """Check if ffmpeg is available in Windows PATH."""
    try:
        # Use 'where.exe' to check for ffmpeg on Windows
        subprocess.run(['where.exe', 'ffmpeg'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_wav_to_mp3(wav_path, quality, delete_wav):
    """Convert a single WAV file to MP3."""
    wav_path = Path(wav_path)
    mp3_path = wav_path.with_suffix('.mp3')
    
    # Skip if MP3 already exists
    if mp3_path.exists():
        print(f"⏭️  Skipping: {mp3_path.name} already exists")
        return True
    
    print(f"🎵 Converting: {wav_path}")
    
    # Build ffmpeg command (list format handles spaces in paths automatically)
    cmd = [
        'ffmpeg',
        '-i', str(wav_path),
        '-codec:a', 'libmp3lame',
        '-qscale:a', quality,
        str(mp3_path),
        '-loglevel', 'error',
        '-stats'
    ]
    
    try:
        # Run ffmpeg
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print(f"✅ Success: {mp3_path}")
        
        # Delete original if requested
        if delete_wav:
            wav_path.unlink()
            print(f"🗑️  Deleted: {wav_path}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Failed to convert {wav_path}")
        if e.stderr:
            print(f"   Details: {e.stderr}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return False

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description='Convert WAV files to MP3 recursively (Windows)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_wav_to_mp3.py
  python convert_wav_to_mp3.py -q 5 -d
  python convert_wav_to_mp3.py --quality 0 --delete

Quality levels:
  0 = highest (~245 kbps)
  2 = high (~190 kbps) - recommended
  5 = medium (~130 kbps)
  9 = lowest (~65 kbps)
        """
    )
    parser.add_argument('-q', '--quality', default=DEFAULT_QUALITY, 
                       help='MP3 quality 0-9 (lower is better, default: 2)')
    parser.add_argument('-d', '--delete', action='store_true', 
                       help='Delete original WAV files after successful conversion')
    args = parser.parse_args()
    
    # Validate quality
    try:
        quality_int = int(args.quality)
        if not 0 <= quality_int <= 9:
            raise ValueError
    except ValueError:
        print(f"Error: Quality must be 0-9, got '{args.quality}'", file=sys.stderr)
        sys.exit(1)
    
    # Check for ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg is not found in PATH.", file=sys.stderr)
        print("\nInstallation instructions for Windows:", file=sys.stderr)
        print("1. Download ffmpeg from: https://ffmpeg.org/download.html", file=sys.stderr)
        print("2. Extract the zip file (e.g., to C:\\ffmpeg)", file=sys.stderr)
        print("3. Add to PATH:", file=sys.stderr)
        print("   - Search Start Menu for 'Environment Variables'", file=sys.stderr)
        print("   - Click 'Edit the system environment variables'", file=sys.stderr)
        print("   - Click 'Environment Variables...'", file=sys.stderr)
        print("   - Under 'System variables', find 'Path' → 'Edit'", file=sys.stderr)
        print("   - Click 'New' and add ffmpeg's 'bin' folder path", file=sys.stderr)
        print("     (e.g., C:\\ffmpeg\\bin)", file=sys.stderr)
        print("4. Restart your terminal", file=sys.stderr)
        sys.exit(1)
    
    print(f"Starting conversion (Quality: {args.quality}, Delete originals: {args.delete})")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # Find all WAV files
    wav_files = list(Path('.').rglob('*.wav'))
    total_files = len(wav_files)
    
    if total_files == 0:
        print("No WAV files found.")
        sys.exit(0)
    
    print(f"Found {total_files} WAV file(s) to process...")
    print("=" * 60)
    
    # Process each file
    for i, wav_path in enumerate(wav_files, 1):
        mp3_path = wav_path.with_suffix('.mp3')
        
        # Skip if MP3 exists
        if mp3_path.exists():
            print(f"[{i}/{total_files}] ⏭️  Skipping: {mp3_path.name}")
            skipped_count += 1
            continue
        
        print(f"[{i}/{total_files}] 🎵 Converting: {wav_path}")
        
        if convert_wav_to_mp3(wav_path, args.quality, args.delete):
            success_count += 1
        else:
            error_count += 1
    
    # Final summary
    print("=" * 60)
    print(f"Conversion complete!")
    print(f"  ✅ Successful: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  ⏭️  Skipped: {skipped_count}")
    print(f"  📊 Total: {total_files}")

if __name__ == '__main__':
    main()
