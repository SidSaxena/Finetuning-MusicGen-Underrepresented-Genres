import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import DatasetDict, Audio, load_from_disk
import random

def create_dataset_from_chunks(chunks_dir, output_dir, audio_column="audio", test_split=0.1, sample_rate=32000):
    """
    Create a HuggingFace dataset from existing audio chunks
    
    Args:
        chunks_dir: Directory containing audio chunk files
        output_dir: Directory to save the dataset and CSV files
        audio_column: Name of the audio column in the dataset
        test_split: Fraction of data to use for testing
        sample_rate: Sample rate to use for the audio files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio files (assuming MP3 format, add more extensions if needed)
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(Path(chunks_dir).rglob(f"*{ext}")))
    
    print(f"Found {len(audio_files)} audio chunk files")
    
    # Create a DataFrame with the file paths
    data = []
    for file_path in tqdm(audio_files, desc="Processing audio chunks"):
        # Convert to absolute path
        abs_path = str(file_path.absolute())
        
        # Get file information
        file_name = file_path.name
        
        # Add to data
        data.append({
            "file_path": abs_path,
            "file_name": file_name,
            audio_column: abs_path
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train and test
    split_idx = int(len(df) * (1 - test_split))
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    # Save CSV files
    train_csv_path = os.path.join(output_dir, "train.csv")
    test_csv_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"Created CSV files:")
    print(f"  - Train: {train_csv_path} ({len(train_df)} files)")
    print(f"  - Test: {test_csv_path} ({len(test_df)} files)")
    
    # Create HuggingFace dataset
    dataset = DatasetDict.from_csv({
        "train": train_csv_path,
        "test": test_csv_path
    })
    
    # Cast audio column to Audio type
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sample_rate))
    
    # Save dataset
    dataset.save_to_disk(os.path.join(output_dir, "dataset"))
    
    print(f"\nCreated HuggingFace dataset at {os.path.join(output_dir, 'dataset')}")
    print(f"You can load this dataset with: dataset = load_from_disk('{os.path.join(output_dir, 'dataset')}')")
    
    # Print example usage code
    print("\nExample usage code:")
    print("```python")
    print("from datasets import load_from_disk")
    print(f"dataset = load_from_disk('{os.path.join(output_dir, 'dataset')}')")
    print("dataset = dataset['train']  # Use only training data")
    print("```")
    
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a HuggingFace dataset from existing audio chunks")
    parser.add_argument("--chunks-dir", "-c", type=str, required=True, help="Directory containing audio chunks")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Output directory for dataset files")
    parser.add_argument("--audio-column", "-a", type=str, default="audio", help="Name of the audio column")
    parser.add_argument("--test-split", "-t", type=float, default=0.1, help="Test split ratio (default: 0.1)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=32000, help="Sample rate (default: 32000)")
    
    args = parser.parse_args()
    
    # Create dataset
    create_dataset_from_chunks(
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
        audio_column=args.audio_column,
        test_split=args.test_split,
        sample_rate=args.sample_rate
    )