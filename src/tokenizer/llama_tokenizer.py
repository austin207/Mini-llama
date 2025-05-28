import sentencepiece as spm

class LlamaTokenizer:
    def __init__(self, model_file="tokenizer/llama.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
    
    def encode(self, text):
        return self.sp.encode(text, out_type=int)
    
    def decode(self, ids):
        return self.sp.decode(ids)
    
    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
