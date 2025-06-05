import sentencepiece as spm
import yaml

with open("configs/tokenizer.yaml", "r") as f:
    config = yaml.safe_load(f)["tokenizer"]

input_file = "data/corpus.txt"
vocab_size = config["vocab_size"]
model_type = config.get("model_type", "bpe")
special_tokens = config.get("special_tokens", [])

user_defined_symbols = ""

spm.SentencePieceTrainer.Train(
    input = input_file,
    model_prefix = "src/tokenizer/llama",
    vocab_size = vocab_size,
    model_type = model_type,
    user_defined_symbols = user_defined_symbols,
    bos_id = 2, eos_id = 3, unk_id = 1, pad_id = 0,
    input_sentence_size = 10000000,  # or even 5_000_000
    shuffle_input_sentence = True
)


print("Tokenizer trained and saved as tokenizer/llama.model and tokenizer/llama.vocab")