from transformers import LlamaTokenizer
import os

# Specify the vocab_file directly
vocab_file_path = os.path.abspath("src/tokenizer/llama_spiece/llama.model")

tokenizer = LlamaTokenizer(
    vocab_file=vocab_file_path,
    use_fast=False,
    legacy=False
)

# Save in HF format
tokenizer.save_pretrained("src/tokenizer/hf_format/")
