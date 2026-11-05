def _codes(packed):
    """uint8 [..., K/2] -> uint8 codes [..., K] (element 2i = low nibble)."""
    lo = packed & 15
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


# Nibble slot i (the old layout's 4i..4i+3) -> bit positions of (m0, m1, m2, sign) in the 32-bit word
# (see convert8 in _CUDA_SRC).
_CODE_POS = {3: (6, 7, 8, 15), 7: (22, 23, 24, 31), 2: (2, 3, 4, 11), 6: (18, 19, 20, 27),
             1: (28, 29, 30, 5), 5: (12, 13, 14, 21), 0: (0, 1, 26, 9), 4: (16, 17, 10, 25)}  # dequant layout


def _kperm_col(p):
    """MOE_KPERM: column of a 64-wide k-stage held at position p = 16 j + 8 hk + 2 t + b."""
    return 16 * ((p >> 1) & 3) + 4 * (p >> 4) + 2 * ((p >> 3) & 1) + (p & 1)


def _repack(w, sf, gated, kperm=False):
    """w: uint8 [E, N, 2048]; sf: e4m3 [E, N, 256] -> uint8 [E, N/128, KS, WBLK].
    kperm (MOE_KPERM): K is permuted inside every 64-wide stage (see _kperm_col): the codes follow it, the scales stay one byte
    per row per block (the lane t of the kernel reads block t), and the gate/up rows of GEMM1 are ordered so that its output
    position p holds the inter column GEMM2 reads at K position p."""
    E, N, _ = w.shape
    n_tiles = N // 128
    dev = w.device
    if gated:
        nt = torch.arange(n_tiles, device=dev).view(n_tiles, 1, 1)
        q = torch.arange(8, device=dev).view(1, 8, 1)
        lr = torch.arange(16, device=dev).view(1, 1, 16)
        icol = 8 * q + (lr & 7)  # position of the row inside the 64 inter columns of the n-tile
        if kperm:
            icol = _kperm_col(icol)
        perm = torch.where(lr < 8, 64 * nt + icol, 4096 + 64 * nt + icol).reshape(-1)
    else:
        perm = torch.arange(N, device=dev)
    out = torch.empty((E, n_tiles, KS, WBLK), dtype=torch.uint8, device=dev)
    sf_u8 = sf.view(torch.uint8)
    shifts = torch.tensor([3, 7, 1, 5, 2, 6, 0, 4], dtype=torch.int32, device=dev)  # (half, hk, b) order below
    pos = torch.tensor([_CODE_POS[int(i)] for i in shifts.tolist()], dtype=torch.int64, device=dev)  # [8, 4]
    for e in range(E):
        c = _codes(w[e][perm]).to(torch.int32)  # [N, 4096]
        if kperm:  # position (j, hk, t, b) of a stage <- natural column 16 t + 4 j + 2 hk + b
            c = c.view(N, KS, 4, 4, 2, 2).permute(0, 1, 3, 4, 2, 5).reshape(N, 4096)
        # rows: [nt, wg, w, half, g]; k: [ks, j, hk, t, b]
        c = c.view(n_tiles, 2, 4, 2, 8, KS, 4, 2, 4, 2)
        # -> [nt, ks, wg, w, g, t, j, half, hk, b]
        c = c.permute(0, 5, 1, 2, 4, 8, 6, 3, 7, 9).reshape(n_tiles, KS, 2, 4, 8, 4, 4, 8)
        # Nibble index for (half, hk, b): i = 3 - (half + 2*hk) + 4*b; flattened order (half, hk, b):
        # (0,0,0)->3 (0,0,1)->7 (0,1,0)->1 (0,1,1)->5 (1,0,0)->2 (1,0,1)->6 (1,1,0)->0 (1,1,1)->4
        # Bit layout read by convert8 (rotate + mask per register): code bit b of the code in nibble slot i goes to
        # bit _CODE_POS[i][b] (b: m0, m1, m2, sign).
        cb = (c.to(torch.int64).unsqueeze(-1) >> torch.arange(4, device=dev)) & 1  # [..., 8, 4]
        words = (cb << pos).sum((-1, -2)) & 0xFFFFFFFF
        wbytes = torch.stack([(words >> (8 * i)) & 0xFF for i in range(4)], dim=-1).to(torch.uint8)
        out[e, :, :, :4096] = wbytes.reshape(n_tiles, KS, 4096)
        s = sf_u8[e][perm].view(n_tiles, 2, 4, 2, 8, KS, 4)  # rows [nt, wg, w, half, g]; cols [ks, j]
        s = s.permute(0, 5, 1, 2, 4, 6, 3).reshape(n_tiles, KS, 512)  # [nt, ks, wg, w, g, j, half]
        out[e, :, :, 4096:] = s
    return out


# --------------------------------------------------------------------------------------
# Attention: FA3 paged FP8 path, the FP32 fallback for non-Hopper devices,
# and the file's own Hopper kernel (validated + tuned in prepare).
# --------------------------------------------------------------------------------------


