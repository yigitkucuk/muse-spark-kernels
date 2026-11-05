def _bind_context(x):
    """A thread that has not used CUDA yet has no current context (its first torch.mm warns and patches that up
    itself): make the device's primary context current on the calling thread. Once per thread; never raises."""
    index = x.device.index if x.device.index is not None else 0
    if getattr(_CTX_TLS, "device", None) == index:
        return
    try:
        if not _CTX_API:
            api = _ct.CDLL("libcuda.so.1")
            p_ctx, p_dev = _ct.POINTER(_ct.c_void_p), _ct.POINTER(_ct.c_int)
            api.cuCtxGetCurrent.argtypes = [p_ctx]
            api.cuCtxGetDevice.argtypes = [p_dev]
            api.cuDeviceGet.argtypes = [p_dev, _ct.c_int]
            api.cuDevicePrimaryCtxRetain.argtypes = [p_ctx, _ct.c_int]
            api.cuCtxSetCurrent.argtypes = [_ct.c_void_p]
            _CTX_API.append(api)
        api = _CTX_API[0]
        ctx, cur = _ct.c_void_p(), _ct.c_int(-1)
        if api.cuCtxGetCurrent(_ct.byref(ctx)) != 0:
            return
        if ctx.value and api.cuCtxGetDevice(_ct.byref(cur)) != 0:
            return
        if not ctx.value or cur.value != index:
            dev, primary = _ct.c_int(), _ct.c_void_p()
            if api.cuDeviceGet(_ct.byref(dev), index) != 0:
                return
            if api.cuDevicePrimaryCtxRetain(_ct.byref(primary), dev.value) != 0:
                return
            if api.cuCtxSetCurrent(primary) != 0:
                return
        _CTX_TLS.device = index
    except Exception:  # noqa: BLE001
        return


def kernel_fn(data, state):
    _bind_context(data["attn_input"])
    st = state
    cfg = data["config"]
    T = cfg["tokens"]
    dev = data["attn_input"].device
    # --- attention ---
    av = _attn_variant(st, cfg)
    use_fmha = _fmha_ok(st, cfg)
    pre = None
    split = not use_fmha and av is not None and T <= PREP_SPLIT_MAX_T and PREP_SPLIT.get("slide" if cfg["window"] else "full", False)
    if split and not PREP_SPLIT_AFTER:
        pre = _attn_prep_split(st, data, cfg)
    ev0 = None
    if split and PREP_SPLIT_AFTER:
        ev0 = torch.cuda.Event()
        ev0.record()
    qkv = _mm(st, data["attn_input"], "qkv")
    if ev0 is not None:
        pre = _attn_prep_split(st, data, cfg, ev0)
    q8 = torch.empty((T, Q_SIZE), dtype=torch.uint8, device=dev)
    _run_qk_prep(st, data, qkv, q8, cfg)
    og = _fmha_attn(st, data, q8, qkv, cfg) if use_fmha else _run_attention(st, data, q8, qkv, cfg, av, pre)
    a = _mm(st, og, "o")
    h = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    maxtiles = _max_tiles(T)
    use_cit = CIT and MOE_MODE_SMALL == "fused" and MOE_MODE_LARGE == "fused" and GATHER_PLAN
    # zb layout: [per-expert counts: LOCAL_EXPERTS | work counter: 1 | tile done: maxtiles
    #             | (CIT: GEMM2 done: maxtiles | head, tail, error, pad: 4)]
    zlen = LOCAL_EXPERTS + 1 + maxtiles + ((maxtiles + 4) if use_cit else 0)
    recompute = T >= BND_CUDA_MIN and BND_RECOMPUTE and T >= BND_RC_MIN_T
    if recompute:  # the first boundary keeps only the branch's rms factor; the second rebuilds the fp32 stream from it
        zb = torch.empty((zlen,), dtype=torch.int32, device=dev)
        r1 = torch.empty((T,), dtype=torch.float32, device=dev)
        st["mod"].boundary_rc(a, data["stream"], st["a1"], st["a1"], st["b1"], st["g_pre_ffn_norm"], data["stream"], h, 1, zb,
                              a, r1, st["a1"], st["b1"])
    elif T >= BND_CUDA_MIN:  # the boundary kernel also zeroes zb (counts | work counter | tile done)
        stream1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
        zb = torch.empty((zlen,), dtype=torch.int32, device=dev)
        st["mod"].boundary(a, data["stream"], st["a1"], st["a1"], st["b1"], st["g_pre_ffn_norm"], stream1, h, False, zb)
    else:
        stream1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
        zb = torch.zeros((zlen,), dtype=torch.int32, device=dev)
        _boundary_kernel[(T,)](a, data["stream"], a, st["a1"], st["b1"], st["g_pre_ffn_norm"], stream1, h,
                               HAS_GPOST=False, H=HIDDEN, num_warps=16)
    # --- router + top-8 on a side stream, concurrent with the pooler GEMM ---
    main = torch.cuda.current_stream()
    side = SIDE_STREAM
    serial = T <= ROUTER_SERIAL_MAX or T >= ROUTER_SERIAL_MIN
    if serial:  # the router GEMM alone on all SMs first: cheaper than slowing the pooler down by sharing SMs with it
        logits = _mm(st, h, "router")
    side.wait_stream(main)
    with torch.cuda.stream(side):
        if not serial:
            logits = _mm(st, h, "router", side=True)  # [T, 512] fp32: hi | lo partial logits
        ids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
        topw = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
        lid = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
        pos = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
        cnt = zb[:LOCAL_EXPERTS]
        cit = None
        if use_cit:  # tok_left (T) | combine queue (T) | row -> token map (rows)
            ctok = torch.empty((2 * T + T * TOP_K + LOCAL_EXPERTS * MAX_PADDED_ROWS_PER_EXPERT,), dtype=torch.int32, device=dev)
            cit = dict(zc=zb[LOCAL_EXPERTS + 1 + maxtiles:], tok=ctok, topw=topw)
        st["mod"].moe_route(logits, st["router_bias"], data["local_start"], ids, topw, lid, pos, cnt,
                            ctok[:T] if use_cit else st["no_zero"], ctok[T:2 * T] if use_cit else st["no_zero"])  # one warp per token
    # --- experts ---
    p = _mm(st, h, "pooler")
    main.wait_stream(side)
    moe = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
    if cit is not None:
        cit["out"] = moe
    ex = _moe_forward(st, p, lid, pos, cnt, T, zb=zb[LOCAL_EXPERTS:], cit=cit)
    y, rrow = ex["y"], ex["rrow"]
    if cit is not None:
        pass  # combined in the fused kernel's tail
    elif CUDA_COMBINE:  # one block per token, PDL (the per-expert scale is 1: st["ones32"])
        st["mod"].moe_combine(y, rrow, lid, topw, st["g_post_expert_norm"], moe)
    elif TRITON_PDL:
        _combine_kernel[(T,)](y, rrow, lid, topw, st["ones32"], st["g_post_expert_norm"], moe, H=D_EXPERT, PDL=True, num_warps=8,
                          launch_pdl=True)
    else:
        _combine_kernel[(T,)](y, rrow, lid, topw, st["ones32"], st["g_post_expert_norm"], moe, H=D_EXPERT,
                              num_warps=COMBINE_NW_SMALL if T <= COMBINE_SMALL_T else 8)
    f = _mm(st, moe, "unpooler")
    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    hidden = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    if recompute:
        st["mod"].boundary_rc(f, data["stream"], st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], stream2,
                              hidden, 2, st["no_zero"], a, r1, st["a1"], st["b1"])
    elif T >= BND_CUDA_MIN:
        st["mod"].boundary(f, stream1, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], stream2, hidden,
                           True, st["no_zero"])
    else:
        _boundary_kernel[(T,)](f, stream1, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"],
                               stream2, hidden, HAS_GPOST=True, H=HIDDEN, num_warps=16)
    return hidden, stream2, ids


# --------------------------------------------------------------------------------------
# Self-tests / micro-benchmarks, printed from prepare() when SELFTEST is on.
# They never run on the scored path; they exist to check the kernels against
# a reference and to time the pieces. All timing goes through _time, which
# flushes L2 before each call the way the benchmark's replays do.
# --------------------------------------------------------------------------------------


