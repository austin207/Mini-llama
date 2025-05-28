import torch
import torch.nn as nn
from .layers import RMSNorm, SwiGLU
from .rope import apply_rope, build_rope_cache

class LlamaDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.norm1 = RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self.max_seq_len = max_seq_len

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _maybe_init_rope(self, device):
        if self.rope_cos is None or self.rope_cos.shape[0] < self.max_seq_len:
            cos, sin = build_rope_cache(self.max_seq_len, self.head_dim, device)
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, x, mask = None):
        B, T, C = x.shape
        device = x.device
        self._maybe_init_rope(device)

        h = self.norm1(x)
        qkv = self.qkv_proj(h)
        q, k, v = qkv.chunk(3, dim = -1)

        q = apply_rope(q, (self.rope_cos[:T], self.rope_sin[:T]))
        k = apply_rope(k, (self.rope_cos[:T], self.rope_sin[:T]))

        attn_scores = torch.einsum('bthd, bshd->bhts', q, k) / (self.head_dim ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim = -1)
        attn_output = torch.einsum('bhts, bshd->bthd', attn_probs, v)
        attn_output =attn_output.reshape(B, T, C)
        attn_output = self.out_proj(attn_output)
        x = x + self.dropout1(attn_output)

        h2 = self.norm2(x)
        mlp_out = self.mlp(h2)
        x = x + self.dropout2(mlp_out)

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
                d_model = config['d_model'],
                n_heads = config['n_heads'],
                d_ff = config['d_ff'],
                max_seq_len = config['max_seq_len'],
                dropout=config['dropout']
            ) for _ in range(config['n_layers'])
        ])
        self.norm = RMSNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], self.vocab_size, bias = False)

    def forward(self, idx, mask = None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, mask = mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits