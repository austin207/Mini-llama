import torch

def build_rope_cache(seq_len, head_dim, device):
    """Build RoPE cache with (seq_len, head_dim//2) shape"""
    assert head_dim % 2 == 0, "head_dim must be even"
    half_dim = head_dim // 2
    theta = 10000.0 ** (-torch.arange(0, half_dim, 1, device=device) / half_dim)
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    freq = pos * theta
    cos = torch.cos(freq)  # (seq_len, half_dim)
    sin = torch.sin(freq)  # (seq_len, half_dim)
    #print(f"[rope.py] Built cache: cos={cos.shape}, sin={sin.shape}")
    return cos, sin

def apply_rope(x, rope_cache):
    """Apply RoPE with proper broadcasting"""
    cos, sin = rope_cache  # (seq_len, half_dim)
    B, T, H, D = x.shape
    half_dim = D // 2
    
    #print(f"\n[rope.py] Applying RoPE to x={x.shape}")
    #print(f"Cache shapes: cos={cos.shape}, sin={sin.shape}")

    # Slice cache to current sequence length
    cos = cos[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, half_dim)
    sin = sin[:T].unsqueeze(0).unsqueeze(2)

    # Split into even/odd indices
    x_even = x[..., :half_dim]
    x_odd = x[..., half_dim:]
    
    #print(f"x_even={x_even.shape}, x_odd={x_odd.shape}")
    #print(f"Broadcasted cos={cos.shape}, sin={sin.shape}")

    # Apply rotation
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    
    # Concatenate rotated features
    x_rotated = torch.cat([x_rot_even, x_rot_odd], dim=-1)
    
    #print(f"Output shape: {x_rotated.shape}")
    return x_rotated
