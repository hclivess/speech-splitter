#!/usr/bin/env python3
"""
Convert HuggingFace Dataset or DatasetDict to Parquet format
Can also convert metadata.csv directly to Parquet
"""

from datasets import load_from_disk, DatasetDict, Dataset
from pathlib import Path
import argparse
import pandas as pd


def dataset_to_parquet(dataset_path: str, output_dir: str, compress: str = "snappy"):
    """
    Convert HuggingFace dataset to Parquet format
    
    Args:
        dataset_path: Path to saved HuggingFace dataset
        output_dir: Directory to save Parquet files
        compress: Compression codec (snappy, gzip, brotli, or None)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Handle both DatasetDict and single Dataset
    if isinstance(dataset, DatasetDict):
        print(f"Found DatasetDict with splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"\nConverting '{split_name}' split ({len(split_data)} samples)...")
            output_file = output_path / f"{split_name}.parquet"
            
            split_data.to_parquet(output_file, compression=compress)
            print(f"✓ Saved to: {output_file}")
    
    else:
        print(f"Converting dataset ({len(dataset)} samples)...")
        output_file = output_path / "dataset.parquet"
        
        dataset.to_parquet(output_file, compression=compress)
        print(f"✓ Saved to: {output_file}")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete! Parquet files saved to: {output_dir}")
    print(f"{'='*60}\n")


def metadata_csv_to_parquet(csv_path: str, output_path: str, compress: str = "snappy"):
    """
    Convert metadata.csv to Parquet
    """
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Converting {len(df)} rows to Parquet...")
    compress_arg = None if compress == 'none' else compress
    df.to_parquet(output_path, compression=compress_arg, index=False)
    
    print(f"✓ Saved to: {output_path}\n")


def base_dir_metadata_to_parquet(base_dir: str, speaker_name: str, output_path: str, compress: str = "snappy"):
    """
    Convert metadata.csv from a speaker directory to Parquet
    Assumes structure: base_dir/speaker_name/metadata.csv
    """
    base_path = Path(base_dir) / speaker_name
    csv_path = base_path / "metadata.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {csv_path}")
    
    print(f"Processing speaker: {speaker_name}")
    print(f"Base directory: {base_dir}")
    metadata_csv_to_parquet(str(csv_path), output_path, compress)


def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets to Parquet format"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # HuggingFace dataset conversion
    hf_parser = subparsers.add_parser(
        "hf",
        help="Convert HuggingFace dataset to Parquet"
    )
    hf_parser.add_argument(
        '--dataset-path',
        required=True,
        help='Path to saved HuggingFace dataset'
    )
    hf_parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for Parquet files'
    )
    hf_parser.add_argument(
        '--compress',
        default='snappy',
        choices=['snappy', 'gzip', 'brotli', 'none'],
        help='Compression codec (default: snappy)'
    )
    
    # Metadata CSV from base directory
    metadata_parser = subparsers.add_parser(
        "metadata",
        help="Convert metadata.csv from speaker directory to Parquet"
    )
    metadata_parser.add_argument(
        '--base-dir',
        required=True,
        help='Base output directory (e.g., C:/ChatterboxTraining/output)'
    )
    metadata_parser.add_argument(
        '--speaker',
        required=True,
        help='Speaker folder name (e.g., "00 Prolog")'
    )
    metadata_parser.add_argument(
        '--output',
        required=True,
        help='Output Parquet file path'
    )
    metadata_parser.add_argument(
        '--compress',
        default='snappy',
        choices=['snappy', 'gzip', 'brotli', 'none'],
        help='Compression codec (default: snappy)'
    )
    
    # Direct CSV conversion
    csv_parser = subparsers.add_parser(
        "csv",
        help="Convert any CSV file to Parquet"
    )
    csv_parser.add_argument(
        '--csv-path',
        required=True,
        help='Path to CSV file'
    )
    csv_parser.add_argument(
        '--output',
        required=True,
        help='Output Parquet file path'
    )
    csv_parser.add_argument(
        '--compress',
        default='snappy',
        choices=['snappy', 'gzip', 'brotli', 'none'],
        help='Compression codec (default: snappy)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "hf":
            compress = None if args.compress == 'none' else args.compress
            dataset_to_parquet(args.dataset_path, args.output_dir, compress)
        
        elif args.command == "metadata":
            compress = None if args.compress == 'none' else args.compress
            base_dir_metadata_to_parquet(args.base_dir, args.speaker, args.output, compress)
        
        elif args.command == "csv":
            compress = None if args.compress == 'none' else args.compress
            metadata_csv_to_parquet(args.csv_path, args.output, compress)
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"\nERROR: Conversion failed!")
        print(f"{e}")
        raise


if __name__ == "__main__":
    main()