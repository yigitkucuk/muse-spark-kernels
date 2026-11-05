def _attention_fallback(data, q8, cfg):
    """Local (non-Hopper) attention over the paged FP8 cache, FP32 math. Testing only: it
    synchronises (`.tolist()`), so it is not on the scored path."""
    B, L = cfg["batch"], cfg["q_len"]
    page, window = cfg["page_size"], cfg["window"]
    ks, vs = data["k_scale"].float(), data["v_scale"].float()
    kf = data["k_cache"].view(-1, HKV, HD)
    vf = data["v_cache"].view(-1, HKV, HD)
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
    out = torch.empty((B * L, HQ, HD), dtype=torch.bfloat16, device=q8.device)
    seq_lens = data["seq_lens"].tolist()
    for b in range(B):
        seq = seq_lens[b]
        first = max(0, seq - window + 1) if window else 0
        kp = torch.arange(first, seq + L, device=q8.device)
        rows = data["page_table"][b].long()[kp // page] * page + kp % page
        keys = kf[rows].float() * ks
        vals = vf[rows].float() * vs
        q = q8[b * L:(b + 1) * L].float()
        for c in range(0, L, FALLBACK_Q_BLOCK):
            qp = seq + torch.arange(c, min(c + FALLBACK_Q_BLOCK, L), device=q8.device)
            vis = kp[None, :] <= qp[:, None]
            if window:
                vis &= kp[None, :] > qp[:, None] - window
            qq = q[c:c + FALLBACK_Q_BLOCK].view(-1, HKV, HQ // HKV, HD)
            att = torch.einsum("lkgd,skd->lkgs", qq, keys) * scale
            att = att.masked_fill(~vis[:, None, None, :], float("-inf")).softmax(-1)
            out[b * L + c: b * L + c + qq.shape[0]] = torch.einsum("lkgs,skd->lkgd", att, vals).reshape(-1, HQ, HD).to(torch.bfloat16)
    return out


def _attention(data, q8, cfg):
    """FA3 paged FP8 attention (or the FP32 fallback when FA3 is unavailable)."""
    if _fa3 is None:
        return _attention_fallback(data, q8, cfg)
    window = cfg["window"]
    scaled = cfg["kv_scale"] != 1.0
    descale = {}
    if scaled:
        B = cfg["batch"]
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, HKV),
                       v_descale=data["v_scale"].reshape(1, 1).expand(B, HKV))
    out = _fa3(q=q8.view(-1, HQ, HD), k_cache=data["k_cache"], v_cache=data["v_cache"],
               page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
               cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=cfg["q_len"],
               softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
               window_size=(window - 1, 0) if window else (-1, -1),
               scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _attention_norm_gate(data, q8, qkv, cfg):
    """FA3 attention followed by the fused `rms(att) * sigmoid(gate)` epilogue -> og [T, Q_SIZE]."""
    T = cfg["tokens"]
    att = _attention(data, q8, cfg)
    og = torch.empty((T, Q_SIZE), dtype=torch.bfloat16, device=q8.device)
    _norm_gate_kernel[(T,)](att, qkv, og, num_warps=4)
    return og


def _attn_splits(cfg, data, variant, nsm):
    """Key-range splits so that small batches still fill the GPU (each split keeps >= 8 key tiles)."""
    B, L, window = cfg["batch"], cfg["q_len"], cfg["window"]
    qb = ATTN_Q_ROWS_PER_WARPGROUP * ATTN_VARIANTS[variant][0]
    base = ((L + qb - 1) // qb) * 2 * B
    if base >= 2 * nsm:
        return 1
    keys = (window + qb) if window else data["page_table"].shape[1] * cfg["page_size"]
    nmax = max(1, min(ATTN_MAX_SPLITS, (keys // ATTN_KEY_TILE) // 8))
    if ATTN_SPLIT_MODE == 1:
        best = None
        for ns in range(1, nmax + 1):
            cost = -(-base * ns // nsm) / ns + (ATTN_SPLIT_PENALTY if ns > 1 else 0.0)
            if best is None or cost < best[0] - 1e-9:
                best = (cost, ns)
        return best[1]
    ns = min((2 * nsm + base - 1) // base, nmax)
    return max(1, min(ns, ATTN_MAX_SPLITS))


def _attn_class(cfg):
    """Shape class of an attention call: the key prepare must choose by."""
    if cfg["window"]:
        return "slide"
    if cfg["tokens"] > 4096:
        return "full_long"
    return "full_batch" if cfg["batch"] > 1 else "full_short"


def _attn_variant(st, cfg):
    """Own-kernel variant for this call's shape class, or None for the FA3 path."""
    choice = st.get("attn_choice")
    if not choice:
        return None
    return choice[_attn_class(cfg)]


def _attn_stage_buffers(data, cfg, dev):
    """Staging buffers for the own attention kernel: K tiles and V tiles per (batch, head)."""
    B, L, window, page = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    smax = (window - 1 + L) if window else data["page_table"].shape[1] * page
    ntk = (smax + ATTN_KEY_TILE - 1) // ATTN_KEY_TILE
    kt = torch.empty((B, 2, ntk, ATTN_KEY_TILE * HD), dtype=torch.uint8, device=dev)
    vt = torch.empty((B, 2, ntk, 72 * ATTN_KEY_TILE), dtype=torch.uint8, device=dev)
    return ntk, kt, vt


def _attn_prep_split(st, data, cfg, ev=None):
    """Stage the tiles made only of keys already in the cache on the side stream (it needs nothing
    from this call's qk_prep, so it overlaps the qkv GEMM's idle SMs); _attn_custom stages the rest
    after joining it. With `ev` (an event recorded before the qkv GEMM was enqueued) the side branch
    is captured after the GEMM node."""
    main, side = torch.cuda.current_stream(), SIDE_STREAM
    pre = _attn_stage_buffers(data, cfg, data["attn_input"].device)
    if ev is not None:
        side.wait_event(ev)
    else:
        side.wait_stream(main)
    with torch.cuda.stream(side):
        st["mod"].attn_prep(data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"],
                            data["seq_lens"], cfg["page_size"], cfg["q_len"], cfg["window"], pre[0], pre[1], pre[2], 1)
    return pre


def _attn_custom(st, data, q8, qkv, cfg, nsplit=1, variant=0, pre=None):
    """FP8 attention (own kernel) with the output norm * sigmoid(gate) fused: returns og [T, Q_SIZE] bf16."""
    B, L, window, page = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    dev = q8.device
    if pre is not None:  # part 1 was issued on the side stream by _attn_prep_split: join it, then stage the rest
        ntk, kt, vt = pre
        torch.cuda.current_stream().wait_stream(SIDE_STREAM)
    else:
        ntk, kt, vt = _attn_stage_buffers(data, cfg, dev)
    st["mod"].attn_prep(data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"],
                        data["seq_lens"], page, L, window, ntk, kt, vt, 2 if pre is not None else 0)
    out = torch.empty((B * L, Q_SIZE), dtype=torch.bfloat16, device=dev)
    c = cfg["sm_scale"] * cfg["qk_scale_factor"] * cfg["kv_scale"] * LOG2E
    if nsplit < 0:  # stream-K over -nsplit CTAs: int header (W, pair prefix, first pairs) + npairs + G partial slots
        G, nqb = -nsplit, (L + ATTN_SK_Q_BLOCK - 1) // ATTN_SK_Q_BLOCK
        hdr = (2 + B * nqb * 2 + 1 + G + 63) // 64 * 64  # must match GLUE's ((2 + npairs + 1 + G) + 63) & ~63
        # Layout agreed with attn_fwd_sk_kernel: header + per-pair blocks of
        # ATTN_SK_Q_BLOCK q-rows x 8 x ATTN_SPLIT_SLOT.
        ws = torch.empty((hdr + (B * nqb * 2 + G) * ATTN_SK_Q_BLOCK * 8 * ATTN_SPLIT_SLOT,), dtype=torch.float32, device=dev)
    else:
        ws = torch.empty((nsplit, B * L, HQ, ATTN_SPLIT_SLOT), dtype=torch.float32, device=dev) if nsplit > 1 else out
    pf = st["o"] if B * L <= ATTN_PF_MAX_T else st["no_pf"]  # o-proj weight into L2 during the attention
    st["mod"].attn_fwd(q8.view(torch.uint8), kt, vt, data["seq_lens"], qkv, out, L, window, ntk, c, cfg["kv_scale"],
                       st["attn_smem"], nsplit, ws, variant, pf)
    return out


def _synthetic_attn_case(B, L, ctx, window, page, seed):
    """A synthetic attention case (random data, paged cache) for validation / tuning."""
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    S = ctx + L
    npg = (S + page - 1) // page
    T = B * L
    q8 = (torch.randn((T, HQ, HD), device=dev, generator=g) * 0.7).to(FP8)
    qkv = torch.randn((T, QKV_SIZE), device=dev, generator=g).to(torch.bfloat16)
    kc = (torch.randn((B * npg, page, HKV, HD), device=dev, generator=g) * 0.8 + 0.3).to(FP8)  # [pages, page_size, 2, 64]
    vc = torch.randn((B * npg, page, HKV, HD), device=dev, generator=g).to(FP8)
    pt = torch.randperm(B * npg, device=dev, generator=g).to(torch.int32).view(B, npg)
    data = {"k_cache": kc, "v_cache": vc, "page_table": pt, "seq_lens": torch.full((B,), ctx, dtype=torch.int32, device=dev),
            "cache_seqlens": torch.full((B,), S, dtype=torch.int32, device=dev),
            "cu_seqlens_q": torch.arange(0, T + 1, L, dtype=torch.int32, device=dev), "scheduler_metadata": None,
            "k_scale": torch.tensor(1.0, device=dev), "v_scale": torch.tensor(1.0, device=dev)}
    cfg = {"batch": B, "q_len": L, "window": window, "page_size": page, "sm_scale": SYNTH_SM_SCALE,
           "qk_scale_factor": SYNTH_QK_SCALE_FACTOR, "kv_scale": 1.0, "tokens": T}
    return data, cfg, q8, qkv


def _attn_ref(data, cfg, q8, qkv):
    """Reference attention path (FA3 + norm-gate) for validating the own kernel."""
    return _attention_norm_gate(data, q8, qkv, cfg)


def _validate_attn(st, variants):
    """Check own-kernel variants against the reference attention path on small synthetic cases; returns
    the variants that agree."""
    worst = {v: 0.0 for v in variants}
    for i, (B, L, ctx, window, page) in enumerate(((2, 300, 1000, 0, 1), (1, 400, 3000, 256, 1), (2, 100, 200, 0, 64))):
        data, cfg, q8, qkv = _synthetic_attn_case(B, L, ctx, window, page, 7 + i)
        ref = _attn_ref(data, cfg, q8, qkv).float()
        for v in variants:
            for ns in (1, 3):
                o = _attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=v).float()
                err = float((o - ref).norm() / ref.norm())
                worst[v] = max(worst[v], err if err == err else 1e9)
    ok = [v for v in variants if worst[v] < ATTN_VALIDATE_TOL]
    print("ATTN own kernel validation (worst rel err):", {v: round(e, 4) for v, e in worst.items()}, "->", ok, flush=True)
    return ok


def _tune_attn(st, variants):
    """Pick, per attention shape class, the fastest of the reference path and the validated own variants."""
    choice = {}
    # The scored shapes themselves (synthetic data): sliding 16K chunk, full 16K chunk at 32K context, turn 1 / turn 4 full.
    for kind, (B, L, ctx, window) in (("slide", (1, 16384, 32768, 2048)), ("full_long", (1, 16384, 32768, 0)),
                                      ("full_short", (1, 753, 80000, 0)), ("full_batch", (4, 753, 80000, 0))):
        data, cfg, q8, qkv = _synthetic_attn_case(B, L, ctx, window, 1, 11)
        res = {"ref": _time(lambda: _attn_ref(data, cfg, q8, qkv), reps=5)}
        for v in variants:
            ns = _attn_splits(cfg, data, v, st["nsm"])
            res[v] = _time(lambda: _attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=v), reps=5)
        best = min(res, key=res.get)
        choice[kind] = None if best == "ref" else best
        print(f"ATTN tune {kind}: " + " ".join(f"{k}={t:.0f}us" for k, t in res.items()) + f" -> {best}", flush=True)
        del data, q8, qkv
    torch.cuda.empty_cache()
    return choice


def _choose_attention(st, props):
    """Validate (and, on Hopper, tune) the own attention kernel; returns the per-class variant choice."""
    if not (props.major >= 9 or OWN_ATTN_LOCAL):
        return None
    try:
        ok = _validate_attn(st, list(ATTN_VARIANTS) if props.major >= 9 else [24])
        if not ok:
            return None
        if props.major >= 9 and ATTN_FIXED is not None and all(v in ok for v in ATTN_FIXED.values()):
            # A prepare-time race between near-equal options adds noise: pin the measured picks.
            return dict(ATTN_FIXED)
        if props.major >= 9:
            return _tune_attn(st, ok)
        return {k: ok[0] for k in ("slide", "full_long", "full_short", "full_batch")}
    except Exception as exc:  # noqa: BLE001
        print("ATTN validation/tuning failed:", repr(exc)[:300], flush=True)
        return None


# --------------------------------------------------------------------------------------
# MoE: gather the pooler output into per-expert rows, run the fused expert
# GEMMs, return the per-route rows for the combine step.
# --------------------------------------------------------------------------------------


