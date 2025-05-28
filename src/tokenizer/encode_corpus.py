import sentencepiece as spm
import numpy as np

# Load your trained tokenizer
sp = spm.SentencePieceProcessor()
sp.load("src/tokenizer/llama.model")

input_file = "data/corpus.txt"
output_file = "data/corpus_ids.npy"
max_seq_len = 256  # Match your model config

all_ids = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        ids = sp.encode(line.strip(), out_type=int)
        # Optionally add <bos> and <eos> tokens if your model expects them
        ids = [sp.bos_id()] + ids + [sp.eos_id()]
        # Chunk into blocks of max_seq_len
        for i in range(0, len(ids), max_seq_len):
            chunk = ids[i:i+max_seq_len]
            if len(chunk) == max_seq_len:
                all_ids.append(chunk)

all_ids = np.array(all_ids, dtype=np.int32)
np.save(output_file, all_ids)
print(f"Saved {all_ids.shape[0]} sequences to {output_file}")
