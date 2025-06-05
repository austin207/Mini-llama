import sentencepiece as smp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import psutil

def get_memory_usage_gb():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def save_checkpoint(all_arrays, checkpoint_dir, checkpoint_num):
    """Save accumulated arrays as a compressed checkpoint"""
    if all_arrays:
        checkpoint_file = f"{checkpoint_dir}/checkpoint_{checkpoint_num:06d}.npz"  # Changed to .npz
        combined_array = np.concatenate(all_arrays)
        
        # Use compressed saving - dramatically reduces disk usage
        np.savez_compressed(checkpoint_file, data=combined_array)
        
        sequences_saved = len(combined_array)
        file_size_mb = os.path.getsize(checkpoint_file) / (1024**2)  # Actual compressed size
        
        print(f"💾 Saved compressed checkpoint {checkpoint_num}: {sequences_saved:,} sequences ({file_size_mb:.1f} MB)")
        return checkpoint_file, sequences_saved
    return None, 0

def process_chunk(chunk_data):
    """Process text chunk with SentencePiece"""
    chunk, tokenizer_path, max_seq_len = chunk_data
    
    # Load SentencePiece model
    sp = smp.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    all_chunks = []
    for text in chunk:
        if text.strip():  # Skip empty lines
            # Tokenize with SentencePiece
            tokens = sp.encode(text.strip(), out_type=int)
            tokens = [sp.bos_id()] + tokens + [sp.eos_id()]  # Add special tokens
            
            # Chunk into sequences
            for i in range(0, len(tokens), max_seq_len):
                chunk_tokens = tokens[i:i+max_seq_len]
                if len(chunk_tokens) >= 128:  # Keep sequences ≥128 tokens
                    # Pad to max_seq_len if needed
                    if len(chunk_tokens) < max_seq_len:
                        chunk_tokens.extend([sp.pad_id()] * (max_seq_len - len(chunk_tokens)))
                    all_chunks.append(chunk_tokens[:max_seq_len])
    
    return np.array(all_chunks, dtype=np.int32) if all_chunks else np.array([], dtype=np.int32).reshape(0, max_seq_len)

def read_file_in_batches(file_path, batch_size):
    """Generator to read file line by line in batches"""
    batch = []
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            batch.append(line.strip())
            line_count += 1
            if len(batch) >= batch_size:
                yield batch, line_count
                batch = []
        # Yield remaining lines
        if batch:
            yield batch, line_count

def combine_checkpoints(checkpoint_files, output_file):
    """Combine all compressed checkpoint files into final output"""
    print(f"🔗 Combining {len(checkpoint_files)} compressed checkpoints into final file...")
    
    all_chunks = []
    total_sequences = 0
    
    for i, checkpoint_file in enumerate(checkpoint_files):
        print(f"📂 Loading compressed checkpoint {i+1}/{len(checkpoint_files)}: {os.path.basename(checkpoint_file)}")
        
        # Load compressed checkpoint
        with np.load(checkpoint_file) as checkpoint_data:
            checkpoint_array = checkpoint_data['data']  # Access the 'data' key
            all_chunks.append(checkpoint_array)
            total_sequences += len(checkpoint_array)
        
        # Memory check
        memory_usage = get_memory_usage_gb()
        print(f"   └─ Loaded {len(checkpoint_array):,} sequences | Memory: {memory_usage:.1f}GB")
    
    # Combine all checkpoints
    print("🔗 Concatenating all checkpoint arrays...")
    final_array = np.concatenate(all_chunks)
    
    # Save final result using compression
    print("💾 Saving final compressed array...")
    np.savez_compressed(output_file, data=final_array)
    
    final_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"✅ Final compressed array saved: {final_array.shape} ({final_size_mb:.1f} MB)")
    
    return final_array

def main():
    # Configuration
    tokenizer_path = "src/tokenizer/llama.model"
    max_seq_len = 1024
    batch_size = 5000  # Lines per batch
    max_workers = min(16, os.cpu_count() - 4)  # Use most CPUs
    checkpoint_dir = "data/checkpoints"
    max_memory_gb = 6  # Reduced memory limit for safety
    max_sequences_per_checkpoint = 80000  # Reduced checkpoint size
    
    print(f"🚀 Starting multiprocessing corpus encoding with COMPRESSED checkpoints...")
    print(f"💾 Workers: {max_workers}")
    print(f"📦 Context length: {max_seq_len} tokens")
    print(f"📊 Batch size: {batch_size} lines")
    print(f"🧠 Memory limit: {max_memory_gb}GB")
    print(f"💾 Checkpoint every: {max_sequences_per_checkpoint:,} sequences")
    print(f"🗜️  Using compressed .npz format to save disk space")
    
    # Setup directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(tokenizer_path):
        print(f"❌ Error: {tokenizer_path} not found!")
        return
    if not os.path.exists("data/corpus.txt"):
        print("❌ Error: data/corpus.txt not found!")
        return
    
    # Get file info and disk space
    file_size_gb = os.path.getsize("data/corpus.txt") / (1024**3)
    disk_free_gb = os.statvfs('.').f_bavail * os.statvfs('.').f_frsize / (1024**3)
    print(f"📊 Corpus file size: {file_size_gb:.2f} GB")
    print(f"💽 Available disk space: {disk_free_gb:.2f} GB")
    
    # Processing variables
    start_time = time.time()
    all_arrays = []
    checkpoint_files = []
    checkpoint_num = 0
    batch_count = 0
    total_sequences = 0
    total_lines_processed = 0
    
    print("📖 Processing corpus with multiprocessing + compressed checkpointing...")
    
    # Process file in batches using multiprocessing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Read and submit batches for processing
        for batch, line_count in read_file_in_batches("data/corpus.txt", batch_size):
            chunk_data = (batch, tokenizer_path, max_seq_len)
            future = executor.submit(process_chunk, chunk_data)
            futures.append(future)
            batch_count += 1
            total_lines_processed = line_count
            
            # Progress update every 100 batches
            if batch_count % 100 == 0:
                elapsed = time.time() - start_time
                lines_per_sec = total_lines_processed / elapsed if elapsed > 0 else 0
                memory_usage = get_memory_usage_gb()
                print(f"🔄 Submitted {batch_count} batches | Lines: {total_lines_processed:,} | "
                      f"Memory: {memory_usage:.1f}GB | Speed: {lines_per_sec:.0f} lines/s")
            
            # Process completed futures periodically
            if len(futures) >= max_workers * 2:
                completed_futures = []
                for i, future in enumerate(futures):
                    if future.done():
                        try:
                            result = future.result()
                            if len(result) > 0:
                                all_arrays.append(result)
                                total_sequences += len(result)
                        except Exception as e:
                            print(f"❌ Error processing batch: {e}")
                        completed_futures.append(i)
                
                # Remove completed futures
                for i in reversed(completed_futures):
                    futures.pop(i)
                
                # Check if we need to save a checkpoint
                memory_usage = get_memory_usage_gb()
                if (memory_usage > max_memory_gb or 
                    total_sequences >= max_sequences_per_checkpoint):
                    
                    # Save compressed checkpoint
                    checkpoint_file, sequences_saved = save_checkpoint(
                        all_arrays, checkpoint_dir, checkpoint_num
                    )
                    if checkpoint_file:
                        checkpoint_files.append(checkpoint_file)
                    
                    # Clear memory
                    all_arrays = []
                    total_sequences = 0
                    checkpoint_num += 1
                    
                    print(f"🧹 Memory cleared | New memory usage: {get_memory_usage_gb():.1f}GB")
        
        # Process remaining futures
        print("🔄 Processing remaining batches...")
        for future in as_completed(futures):
            try:
                result = future.result()
                if len(result) > 0:
                    all_arrays.append(result)
                    total_sequences += len(result)
            except Exception as e:
                print(f"❌ Error processing batch: {e}")
    
    # Save final checkpoint if there are remaining arrays
    if all_arrays:
        checkpoint_file, sequences_saved = save_checkpoint(
            all_arrays, checkpoint_dir, checkpoint_num
        )
        if checkpoint_file:
            checkpoint_files.append(checkpoint_file)
    
    # Combine all checkpoints into final file
    if checkpoint_files:
        final_output = "data/corpus_ids.npz"  # Changed to .npz
        final_array = combine_checkpoints(checkpoint_files, final_output)
        
        # Calculate final stats
        elapsed = time.time() - start_time
        sequences_per_sec = len(final_array) / elapsed if elapsed > 0 else 0
        
        print(f"\n✅ SUCCESS! Processing complete!")
        print(f"📊 Final shape: {final_array.shape}")
        print(f"⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"📈 Lines processed: {total_lines_processed:,}")
        print(f"🚀 Speed: {sequences_per_sec:.0f} sequences/second")
        print(f"💾 Output file: {final_output}")
        print(f"🗜️  Compressed file size: {os.path.getsize(final_output)/(1024**2):.1f} MB")
        
        # Cleanup checkpoint files
        print("🧹 Cleaning up checkpoint files...")
        for checkpoint_file in checkpoint_files:
            os.remove(checkpoint_file)
        os.rmdir(checkpoint_dir)
        print("✅ Cleanup complete!")
        
    else:
        print("❌ No valid sequences found!")

if __name__ == "__main__":
    main()
