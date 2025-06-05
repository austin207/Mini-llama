import numpy as np

# Load your tokenized sequences
sequences = np.load('data/corpus_ids.npy')

# Number of sequences is the first dimension
num_sequences = len(sequences)
print(f"Total sequences: {num_sequences}")

# You can also check the full shape
print(f"Shape: {sequences.shape}")
# Output example: Shape: (50000, 256) means 50,000 sequences of 256 tokens each
