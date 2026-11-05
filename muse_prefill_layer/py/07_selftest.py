def _routes_from_counts(T, counts, dev):
    """Build (lid, pos, cnt) for a synthetic routing: counts[e] tokens to expert e."""
    lid = torch.full((T, TOP_K), -1, dtype=torch.int32, device=dev)
    pos = torch.zeros((T, TOP_K), dtype=torch.int32, device=dev)
    r = 0
    for e, c in enumerate(counts):
        idx = torch.arange(r, r + c, device=dev)
        lid.view(-1)[(idx % T) * TOP_K + idx // T] = e
        pos.view(-1)[(idx % T) * TOP_K + idx // T] = torch.arange(c, device=dev, dtype=torch.int32)
        r += c
    cnt = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device=dev)
    cnt[: len(counts)] = torch.tensor(counts, dtype=torch.int32, device=dev)
    return lid, pos, cnt


_FLUSH_BUF = None


def _time(fn, reps=5):
    """Median time in us with a cold L2 (a read sweep over 256 MB before each call, like the benchmark's replays)."""
    global _FLUSH_BUF
    if _FLUSH_BUF is None:
        _FLUSH_BUF = (torch.ones(128 << 20, dtype=torch.float16, device="cuda"),
                      torch.zeros((), dtype=torch.float32, device="cuda"))
    buf, sink = _FLUSH_BUF
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        sink.copy_(buf.sum(dtype=torch.float32))
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


SCEN = {
    "T16384": (16384, [12547, 11910, 2820, 2392, 270, 242, 97, 60, 34, 16, 14, 2, 1, 1, 1]),
    "T3012": (3012, [2277, 2191, 416, 400, 47, 37, 32, 11, 7, 5, 4, 3, 2, 1, 1]),
    "T753": (753, [540, 529, 127, 108, 14, 6, 3, 3, 3, 2, 1, 1]),
}


def _selftest(st, weights):
    from nvfp4 import dequantize
    dev = weights["w13"].device
    mod = st["mod"]
    torch.manual_seed(0)
    # correctness on a small skewed routing
    T = 700
    counts = [300, 200, 100, 60, 33, 17, 14, 5, 1]  # 14 -> N=16, 5 and 1 -> N=8 tiles (below FINE_MIN_T)
    lid, pos, cnt = _routes_from_counts(T, counts, dev)
    p = torch.randn((T, D_EXPERT), device=dev).to(torch.bfloat16)
    gw = st["g_pre_expert_norm"]
    out = _moe_forward(st, p, lid, pos, cnt, T)
    y, rrow = out["y"], out["rrow"]
    pf = p.float()
    xn = (pf * torch.rsqrt(pf.pow(2).mean(-1, keepdim=True) + RMS_EPS) * (1.0 + gw.float())).to(torch.bfloat16)
    msg = []
    for e in range(len(counts)):
        tok, slot = (lid == e).nonzero(as_tuple=True)
        W13 = dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).to(torch.bfloat16)
        W2 = dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).to(torch.bfloat16)
        hh = xn[tok].float() @ W13.float().T
        inter = (F.silu(hh[:, :D_EXPERT]) * hh[:, D_EXPERT:]).to(torch.bfloat16)
        ref = inter.float() @ W2.float().T
        got = y[rrow[tok, slot].long()].float()
        msg.append(f"e{e}:n{len(tok)}:{float((got - ref).norm() / ref.norm()):.4f}")
    print("SELFTEST moe rel err", " ".join(msg), flush=True)
    # timing
    for name, (T, counts) in SCEN.items():
        lid, pos, cnt = _routes_from_counts(T, counts, dev)
        p = torch.randn((T, D_EXPERT), device=dev).to(torch.bfloat16)
        out = _moe_forward(st, p, lid, pos, cnt, T)
        nt = int(out["ntiles"])
        rows = int(out["tiles"][:nt, 2].sum())
        R = sum(counts)
        res = []
        P1, P2 = 8, 16
        t1 = _time(lambda: mod.moe_gemm(st["w13r"], out["xt"], out["tiles"], out["ntiles"], st["w13_gs"], out["inter"], 64, P1,
                                        True, st["smem"]))
        t2 = _time(lambda: mod.moe_gemm(st["w2r"], out["inter"], out["tiles"], out["ntiles"], st["w2_gs"], out["y"], 32, P2,
                                        False, st["smem"]))
        f1 = 2 * R * (2 * D_EXPERT) * D_EXPERT / t1 / 1e6  # GEMM1: [R, D_EXPERT] x [2*D_EXPERT, D_EXPERT]^T
        f2 = 2 * R * D_EXPERT * D_EXPERT / t2 / 1e6  # GEMM2: [R, D_EXPERT] x [D_EXPERT, D_EXPERT]^T
        res.append(f"P{P1}/{P2}: g1 {t1:.0f}us ({f1:.0f}TF) g2 {t2:.0f}us ({f2:.0f}TF)")
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        rr = out["rrow"][out["rrow"] >= 0].long()
        y0 = out["y"][rr].clone()
        ctr = torch.zeros((2,), dtype=torch.int32, device=dev)
        rows_t = torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)

        def g1p():
            ctr.zero_()
            mod.moe_gemm_p(st["w13r"], out["xt"], out["tiles"], out["ntiles"], st["w13_gs"], out["inter"], 64, nsm, True,
                           st["smem"], ctr[0:1])

        def g2p():
            ctr.zero_()
            mod.moe_gemm_p(st["w2r"], out["inter"], out["tiles"], out["ntiles"], st["w2_gs"], out["y"], 32, nsm, False,
                           st["smem"], ctr[1:2])
        zb = torch.zeros((1 + out["tiles"].shape[0],), dtype=torch.int32, device=dev)

        def gf():
            zb.zero_()
            mod.moe_fused(st["w13r"], st["w2r"], out["xt"], out["inter"], out["y"], out["tiles"], out["ntiles"], st["w13_gs"],
                          st["w2_gs"], nsm, st["smem"], zb[0:1], zb[1:], st["moe_dbg"], st["moe_flags"], st["no_zero"],
                          st["no_zero"], out["rrow"], lid, st["no_zero_f"], st["g_post_expert_norm"], st["no_zero_b"])
        for order in (0, 1, 2):
            mod.moe_plan(cnt, out["offsets"], out["tiles"], out["ntiles"], rows_t, order)
            tf = _time(gf)
            same_f = bool(torch.equal(out["y"][rr], y0))
            res.append(f"fused o{order}: {tf:.0f}us same={same_f}")
            t1 = _time(g1p)
            t2 = _time(g2p)
            f1 = 2 * R * 8192 * 4096 / t1 / 1e6
            f2 = 2 * R * 4096 * 4096 / t2 / 1e6
            same = bool(torch.equal(out["y"][rr], y0))
            res.append(f"persist o{order}: g1 {t1:.0f}us ({f1:.0f}TF) g2 {t2:.0f}us ({f2:.0f}TF) same={same}")
        tg = _time(lambda: mod.moe_gather(p, st["g_pre_expert_norm"], lid, pos, out["offsets"], cnt, out["xt"], out["rrow"]))
        print(f"SELFTEST {name} R={R} padded={rows} tiles={nt} gather {tg:.0f}us |", " | ".join(res), flush=True)
    a = torch.randn((16384, 8192), device=dev).to(torch.bfloat16)
    tmm = _time(lambda: torch.mm(a, st["pooler"].t()))
    print(f"SELFTEST cublas 16384x4096x8192 {tmm:.0f}us {2 * 16384 * D_EXPERT * HIDDEN / tmm / 1e6:.0f}TF", flush=True)
    s = torch.randn((16384, 8192), device=dev)
    o1 = torch.empty_like(s)
    o2 = torch.empty((16384, 8192), device=dev, dtype=torch.bfloat16)
    for nw in (4, 8, 16):
        tb = _time(lambda: _boundary_kernel[(16384,)](a, s, st["g_post_ffn_norm"], st["a2"], st["b2"],
                                             st["g_next_pre_attn_norm"], o1, o2, HAS_GPOST=True, H=HIDDEN, num_warps=nw))
        print(f"SELFTEST boundary nw{nw} {tb:.0f}us {16384 * HIDDEN * 12 / tb / 1e6:.2f}TB/s", flush=True)
    del a, s, o1, o2
    torch.cuda.empty_cache()


ATTN_SCEN = [
    ("turn1_full", 1, 753, 80000, 0),
    ("turn4_full", 4, 753, 80000, 0),
    ("ft_full32k", 1, 16384, 32768, 0),
    ("ft_full64k", 1, 16384, 65536, 0),
    ("ft_slide", 1, 16384, 32768, 2048),
]


def _selftest_attn2(st):
    if _fa3 is None:
        return
    dev = "cuda"
    torch.manual_seed(2)
    for name, B, L, ctx, window in ATTN_SCEN:
        S = ctx + L
        T = B * L
        q8 = (torch.randn((T, HQ, HD), device=dev) * 0.7).to(FP8)
        qkv = torch.randn((T, QKV_SIZE), device=dev).to(torch.bfloat16)
        kc = (torch.randn((B * S, 1, HKV, HD), device=dev) * 0.8 + 0.4).to(FP8)
        vc = torch.randn((B * S, 1, HKV, HD), device=dev).to(FP8)
        perm = torch.randperm(B * S, device=dev).to(torch.int32)
        data = {"k_cache": kc, "v_cache": vc, "page_table": perm.view(B, S), "seq_lens": torch.full((B,), ctx, dtype=torch.int32, device=dev),
                "cache_seqlens": torch.full((B,), S, dtype=torch.int32, device=dev),
                "cu_seqlens_q": torch.arange(0, T + 1, L, dtype=torch.int32, device=dev), "scheduler_metadata": None,
                "k_scale": torch.tensor(1.0, device=dev), "v_scale": torch.tensor(1.0, device=dev)}
        cfg = {"batch": B, "q_len": L, "window": window, "page_size": 1, "sm_scale": SYNTH_SM_SCALE,
               "qk_scale_factor": SYNTH_QK_SCALE_FACTOR, "kv_scale": 1.0, "tokens": T}

        def ref_fn():
            return _attention_norm_gate(data, q8, qkv, cfg)
        flops = 4.0 * HQ * HD * (B * L * (ctx if not window else min(ctx, window)) + B * L * L / 2)
        try:
            r = ref_fn()
            t_ref = _time(ref_fn, reps=3)
            res = [f"fa3+ng {t_ref:.0f}us ({flops / t_ref / 1e6:.0f}TF)"]
        except Exception as exc:  # noqa: BLE001
            print(f"SELFTEST attn2 {name} FA3 ERR {repr(exc)[:300]}", flush=True)
            continue
        for variant, nsplit in [(v, 1) for v in ATTN_VARIANTS] + [(v, _attn_splits(cfg, data, v, st["nsm"])) for v in (1, 9)]:
            try:
                o = _attn_custom(st, data, q8, qkv, cfg, nsplit=nsplit, variant=variant)
                torch.cuda.synchronize()
                err = float((o.float() - r.float()).norm() / r.float().norm())
                t_own = _time(lambda: _attn_custom(st, data, q8, qkv, cfg, nsplit=nsplit, variant=variant), reps=3)
                res.append(f"v{variant}s{nsplit} {t_own:.0f}us ({flops / t_own / 1e6:.0f}TF) err {err:.4f}")
            except Exception as exc:  # noqa: BLE001
                res.append(f"v{variant}s{nsplit} ERR {repr(exc)[:120]}")
        print(f"SELFTEST attn2 {name}: " + " | ".join(res), flush=True)
        del q8, qkv, kc, vc
    torch.cuda.empty_cache()


def _selftest_boundary(st):
    dev = "cuda"
    T = 16384
    a = torch.randn((T, HIDDEN), device=dev).to(torch.bfloat16)
    s = torch.randn((T, HIDDEN), device=dev)
    o1 = torch.empty_like(s)
    o2 = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    o3 = torch.empty_like(s)
    o4 = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    _boundary_kernel[(T,)](a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], o1, o2,
                           HAS_GPOST=True, H=HIDDEN, num_warps=16)
    st["mod"].boundary(a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], o3, o4, True, st["no_zero"])
    torch.cuda.synchronize()
    d1 = float((o1 - o3).abs().max())
    d2 = float((o2.float() - o4.float()).abs().max())
    t1 = _time(lambda: _boundary_kernel[(T,)](a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"],
                                            o1, o2, HAS_GPOST=True, H=HIDDEN, num_warps=16))
    t2 = _time(lambda: st["mod"].boundary(a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"],
                                         o3, o4, True, st["no_zero"]))
    print(f"SELFTEST boundary triton {t1:.0f}us cuda {t2:.0f}us (diff {d1:.1e} {d2:.1e})", flush=True)
    del a, s, o1, o2, o3, o4
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Extension build + CUDA / C++ sources. PRUNED (v2): these five strings
# are the tuned algorithm with every dead branch removed: all
# preprocessor branches not taken under either real build config
# (__CUDA_ARCH__ = 900 / sm_90a, no -D flags from _build, in-file
# macro defaults, experiment macros undefined, MOE_FAST_SCALE in
# {0, 1} -- the one switch _build itself supplies, 0 when the weights
# have non-normal block scales), every function / define / variant
# unreachable from the exported functions, and the dead comments
# attached to them. The taken path is unchanged (checked by
# preprocessing v1 and v2 under both configs: identical except for
# the removed dead items). They hold the grouped NVFP4 expert GEMMs,
# the boundary / route / combine / qk-prep kernels, the own Hopper
# attention kernel, and the cuBLASLt glue. Any change to a taken-path
# instruction here is a numerics or performance change, not a cleanup,
# and needs its own GPU gating.
# The Python constants that must agree with defines inside _CUDA_SRC are
# marked at their definitions (ATTN_ASTAGES, ATTN_SMEM).
# --------------------------------------------------------------------------------------


