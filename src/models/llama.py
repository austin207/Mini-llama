import torch
import torch.nn as nn
from .layers import RMSNorm, SwiGLU
from .rope import apply_rope, build_rope_cache

class LlamaDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout):
        super().__init__()
        assert d_model % n_heads == 0, f"{d_model} must be divisible by {n_heads}"
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Components
        self.norm1 = RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

        # RoPE cache
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def _init_rope(self, device):
        """Initialize RoPE cache with CORRECT dimensions"""
        if self.rope_cos is None or self.rope_cos.size(1) < self.max_seq_len:
            self.rope_cos, self.rope_sin = build_rope_cache(
                seq_len=self.max_seq_len,
                head_dim=self.head_dim,
                device=device
            )
            #print(f"\n[llama.py] Initialized RoPE cache:")
            #print(f"cos: {self.rope_cos.shape}, sin: {self.rope_sin.shape}")

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        device = x.device
        self._init_rope(device)

        # --- Attention ---
        h = self.norm1(x)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        
        # Reshape to (B, T, H, D)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        #print(f"\n[llama.py] Before RoPE:")
        #print(f"q: {q.shape}, k: {k.shape}")
        #print(f"cache: {self.rope_cos.shape}, {self.rope_sin.shape}")

        # Apply RoPE with sliced cache
        q = apply_rope(q, (self.rope_cos, self.rope_sin))
        k = apply_rope(k, (self.rope_cos, self.rope_sin))

        #print(f"\n[llama.py] After RoPE:")
        #print(f"q: {q.shape}, k: {k.shape}")

        # Attention scores
        attn_scores = torch.einsum('bthd,bshd->bhts', q, k) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum('bhts,bshd->bthd', attn_probs, v)
        
        # Merge heads and project
        attn_output = attn_output.reshape(B, T, self.d_model)
        attn_output = self.out_proj(attn_output)
        x = x + self.dropout(attn_output)

        # --- MLP ---
        h = self.norm2(x)
        mlp_out = self.mlp(h)
        x = x + self.dropout(mlp_out)

        return x

class MiniLlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.max_seq_len = config['max_seq_len']

        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.blocks = nn.ModuleList([
            LlamaDecoderBlock(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                max_seq_len=config['max_seq_len'],
                dropout=config.get('dropout', 0.1)
            ) for _ in range(config['n_layers'])
        ])
        self.norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def forward(self, idx, mask=None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
