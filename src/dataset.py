import numpy as np
import torch
import os

class LlamaDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, max_sequences=None, sequence_length=1024):
        """
        Enhanced dataset that handles both .npy and .dat files with subset training
        
        Args:
            data_file: Path to .npy or .dat file
            max_sequences: Optional limit for subset training (None = use all)
            sequence_length: Length of each sequence (for .dat files)
        """
        self.data_file = data_file
        self.sequence_length = sequence_length
        
        print(f"📂 Loading dataset from {data_file}")
        
        if data_file.endswith('.npy'):
            # Original .npy file handling
            self.data = np.load(data_file)
            print(f"📊 Loaded .npy file: {len(self.data):,} sequences")
            
        elif data_file.endswith('.dat'):
            # Memory-mapped .dat file handling (your concatenated file)
            file_size = os.path.getsize(data_file)
            bytes_per_sequence = sequence_length * 4  # int32 = 4 bytes
            total_sequences = file_size // bytes_per_sequence
            
            self.data = np.memmap(
                data_file, 
                dtype=np.int32, 
                mode='r',
                shape=(total_sequences, sequence_length)
            )
            print(f"📊 Loaded .dat file: {total_sequences:,} sequences")
            
        else:
            raise ValueError(f"Unsupported file format: {data_file}. Use .npy or .dat")
        
        # Apply subset limit if specified
        if max_sequences and max_sequences < len(self.data):
            self.data = self.data[:max_sequences]
            print(f"🎯 Using subset: {len(self.data):,} sequences (limited from {max_sequences:,})")
        
        print(f"✅ Dataset ready: {len(self.data):,} sequences")
        print(f"📈 Total tokens: {len(self.data) * (sequence_length-1):,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Same interface as before - input/target splitting
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)  # Input: all but last token
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)   # Target: all but first token
        return x, y
