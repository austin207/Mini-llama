import torch

def apply_rope(x, rope_cahe):
    cos, sin = rope_cahe
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rope = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim = -1)
    return x_rope

def build_rope_cache(seq_len, head_dim, device):
    theta = 10000.0 ** (-torch.arange(0, head_dim, 2, device = device) / head_dim)
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    freq = pos * theta
    cos = torch.cos(freq)
    sin = torch.sin(freq)

    cos = torch.repeat_interleave(cos, 2, dim = 1)
    sin = torch.repeat_interleave(sin, 2, dim = 1)
    return cos, sin