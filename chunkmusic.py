import os
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm
import random
import json
import shutil

def create_audio_chunks(input_dir, output_dir, chunk_duration=30, overlap=0, min_duration=10, 
                        format="mp3", sample_rate=32000, metadata_file="metadata.csv"):
    """
    Split audio files into chunks for dataset creation
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save chunks
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds
        min_duration: Minimum duration for a valid chunk in seconds
        format: Output audio format (mp3, wav)
        sample_rate: Output sample rate
        metadata_file: File to save metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio files
    audio_extensions = ['.mp3', '.wav', '.m4a', '.webm', '.flac', '.ogg']
    audio_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Prepare metadata collection
    metadata = []
    
    # Process each file
    for file_idx, file_path in enumerate(tqdm(audio_files, desc="Processing files")):
        try:
            # Get file metadata
            filename = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(filename)[0]
            
            # Load audio with pydub for easy slicing
            audio = AudioSegment.from_file(file_path)
            
            # Convert to milliseconds
            chunk_ms = chunk_duration * 1000
            overlap_ms = overlap * 1000
            min_duration_ms = min_duration * 1000
            audio_duration_ms = len(audio)
            
            if audio_duration_ms < min_duration_ms:
                print(f"Skipping {filename}: too short ({audio_duration_ms/1000:.1f}s)")
                continue
            
            # Calculate number of chunks
            step_ms = chunk_ms - overlap_ms
            num_chunks = max(1, int((audio_duration_ms - overlap_ms) / step_ms))
            
            # Create chunks
            for chunk_idx in range(num_chunks):
                start_ms = chunk_idx * step_ms
                end_ms = min(start_ms + chunk_ms, audio_duration_ms)
                
                # Skip if the chunk is too short
                if end_ms - start_ms < min_duration_ms:
                    continue
                
                # Extract chunk
                chunk = audio[start_ms:end_ms]
                
                # Create output filename
                chunk_filename = f"{file_name_without_ext}_chunk{chunk_idx+1:03d}.{format}"
                chunk_path = os.path.join(output_dir, chunk_filename)
                
                # Export chunk
                chunk.export(
                    chunk_path, 
                    format=format,
                    bitrate=f"256k",
                    parameters=["-ar", str(sample_rate)]
                )
                
                # Add to metadata
                metadata.append({
                    "file_id": f"{file_idx+1:04d}_{chunk_idx+1:03d}",
                    "original_file": filename,
                    "chunk_file": chunk_filename,
                    "start_time": start_ms / 1000,
                    "end_time": end_ms / 1000,
                    "duration": (end_ms - start_ms) / 1000,
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save metadata
    if metadata:
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(output_dir, metadata_file), index=False)
        print(f"Created {len(metadata)} chunks")
        print(f"Metadata saved to {os.path.join(output_dir, metadata_file)}")
    else:
        print("No valid chunks were created.")

def create_huggingface_dataset(input_dir, output_dir, metadata_file="metadata.csv", test_split=0.1):
    """
    Convert chunked audio files to a HuggingFace dataset format
    
    Args:
        input_dir: Directory containing audio chunks
        output_dir: Directory to save the formatted dataset
        metadata_file: CSV file with metadata
        test_split: Fraction of data to use for testing
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # Read metadata
    metadata_path = os.path.join(input_dir, metadata_file)
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return
    
    metadata_df = pd.read_csv(metadata_path)
    
    # Split into train and test
    all_files = metadata_df["chunk_file"].unique().tolist()
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * (1 - test_split))
    train_files = set(all_files[:split_idx])
    test_files = set(all_files[split_idx:])
    
    # Prepare dataset entries
    train_data = []
    test_data = []
    
    # For each chunk
    for _, row in tqdm(metadata_df.iterrows(), desc="Creating dataset", total=len(metadata_df)):
        chunk_file = row["chunk_file"]
        src_path = os.path.join(input_dir, chunk_file)
        
        # Skip if file doesn't exist
        if not os.path.exists(src_path):
            continue
        
        # Decide if this goes to train or test
        if chunk_file in train_files:
            target_dir = os.path.join(output_dir, "train")
            target_data = train_data
        else:
            target_dir = os.path.join(output_dir, "test")
            target_data = test_data
        
        # Copy the file
        target_path = os.path.join(target_dir, chunk_file)
        shutil.copy2(src_path, target_path)
        
        # Add metadata
        target_data.append({
            "file": chunk_file,
            "audio": chunk_file,
            "original_file": row["original_file"],
            "duration": row["duration"],
        })
    
    # Save dataset metadata
    with open(os.path.join(output_dir, "train", "metadata.jsonl"), 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
            
    with open(os.path.join(output_dir, "test", "metadata.jsonl"), 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created dataset with {len(train_data)} training samples and {len(test_data)} test samples")
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create audio chunks dataset from music files")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Input directory with audio files")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Output directory for chunks")
    parser.add_argument("--chunk-duration", "-d", type=int, default=30, help="Duration of each chunk in seconds (default: 30)")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks in seconds (default: 0)")
    parser.add_argument("--min-duration", type=int, default=10, help="Minimum duration for a chunk (default: 10)")
    parser.add_argument("--format", "-f", type=str, default="mp3", help="Output audio format (default: mp3)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=32000, help="Output sample rate (default: 32000)")
    parser.add_argument("--huggingface", "-hf", action="store_true", help="Create HuggingFace dataset structure")
    parser.add_argument("--test-split", "-ts", type=float, default=0.1, help="Test split ratio (default: 0.1)")
    
    args = parser.parse_args()
    
    # Create chunks
    create_audio_chunks(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        min_duration=args.min_duration,
        format=args.format,
        sample_rate=args.sample_rate
    )
    
    # Optionally create HuggingFace dataset
    if args.huggingface:
        hf_output_dir = args.output_dir + "_hf"
        create_huggingface_dataset(
            input_dir=args.output_dir,
            output_dir=hf_output_dir,
            test_split=args.test_split
        )