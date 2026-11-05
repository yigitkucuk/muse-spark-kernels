def _max_tiles(T):
    """Upper bound on the plan's tile count for T tokens (min-size tiles + slack)."""
    return T * TOP_K // MIN_TILE_ROWS + MOE_TILE_SLACK


def _moe_forward(st, p, lid, pos, cnt, T, order=None, zb=None, cit=None):
    """Routed experts on the pooler output p [T, 4096]: returns per-route rows y [rows, 4096] (bf16) and rrow.
    With cit (dict: zc = [max tiles | 4] zeroed counters, tok = [2T + rows] ints, topw, out) the fused kernel
    also writes the combined output (combine in the tail) and the caller skips the combine kernel."""
    dev = p.device
    mod = st["mod"]
    maxrows = T * TOP_K + LOCAL_EXPERTS * MAX_PADDED_ROWS_PER_EXPERT
    maxtiles = _max_tiles(T)
    offsets = torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)
    rows = torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)
    tiles = torch.empty((maxtiles, 4), dtype=torch.int32, device=dev)
    ntiles = torch.empty((1,), dtype=torch.int32, device=dev)
    large = T >= MOE_LARGE_T
    mode = MOE_MODE_LARGE if large else MOE_MODE_SMALL
    if order is None:
        order = MOE_ORDER_LARGE if large else MOE_ORDER_SMALL
    fine = 1 if (FINE_MIN_T and T >= FINE_MIN_T) else 0  # remainder tiles rounded to 32 rows (plan + kernel agree)
    order = order | (fine << MOE_FINE_ORDER_SHIFT)
    xt = torch.empty((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
    rrow = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    if cit is not None and (not GATHER_PLAN or mode != "fused"):
        raise RuntimeError("CIT needs the fused kernel and the gather+plan launch")
    cz = cit["zc"] if cit is not None else st["no_zero"]
    ctok = cit["tok"] if cit is not None else st["no_zero"]
    if GATHER_PLAN:  # the gather kernel also writes the plan (one launch fewer)
        mod.moe_gather_plan(p, st["g_pre_expert_norm"], lid, pos, cnt, xt, rrow, offsets, tiles, ntiles, rows, order, cz, ctok)
    else:
        mod.moe_plan(cnt, offsets, tiles, ntiles, rows, order)
        mod.moe_gather(p, st["g_pre_expert_norm"], lid, pos, offsets, cnt, xt, rrow)
    inter = torch.empty((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
    y = torch.empty((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
    if mode == "fused":
        if zb is None:
            zb = torch.zeros((1 + maxtiles,), dtype=torch.int32, device=dev)  # work counter + per-tile GEMM1 completion
        mod.moe_fused(st["w13r"], st["w2r"], xt, inter, y, tiles, ntiles, st["w13_gs"], st["w2_gs"],
                      st["nsm"] * (2 if SKINNY_TEST else 1), st["smem"],
                      zb[0:1], zb[1:1 + maxtiles], st["moe_dbg"], st["moe_flags"] | (fine << MOE_FINE_ORDER_SHIFT), cz, ctok,
                      rrow, lid, cit["topw"] if cit is not None else st["no_zero_f"], st["g_post_expert_norm"],
                      cit["out"] if cit is not None else st["no_zero_b"])
    elif mode == "persist":
        ctr = torch.zeros((2,), dtype=torch.int32, device=dev)
        mod.moe_gemm_p(st["w13r"], xt, tiles, ntiles, st["w13_gs"], inter, 64, st["nsm"], True, st["smem"], ctr[0:1])
        mod.moe_gemm_p(st["w2r"], inter, tiles, ntiles, st["w2_gs"], y, 32, st["nsm"], False, st["smem"], ctr[1:2])
    else:
        mod.moe_gemm(st["w13r"], xt, tiles, ntiles, st["w13_gs"], inter, 64, P_GEMM1, True, st["smem"])
        mod.moe_gemm(st["w2r"], inter, tiles, ntiles, st["w2_gs"], y, 32, P_GEMM2, False, st["smem"])
    return dict(y=y, rrow=rrow, xt=xt, inter=inter, tiles=tiles, ntiles=ntiles, offsets=offsets)


# --------------------------------------------------------------------------------------
# kernel_fn: one prefill layer step.
# --------------------------------------------------------------------------------------


def _run_qk_prep(st, data, qkv, q8, cfg):
    """Q/K RMS-norm + RoPE + FP8 quantisation + cache stores (CUDA kernel, or the Triton twin)."""
    if QK_PREP_CUDA:
        st["mod"].qk_prep(qkv, data["positions"], data["cache_loc"], st["inv_freq"], q8, data["k_cache"].view(torch.uint8),
                          data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], bool(cfg["rope_theta"]),
                          cfg["kv_scale"] != 1.0)
    else:
        T = cfg["tokens"]
        _qk_prep_kernel[(T,)](qkv, data["positions"], data["cache_loc"], st["inv_freq"], q8,
                              data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"],
                              data["v_scale"], ROPE=bool(cfg["rope_theta"]), SCALED=cfg["kv_scale"] != 1.0, num_warps=4,
                              **(dict(PDL=True, launch_pdl=True) if TRITON_PDL else {}))


def _run_attention(st, data, q8, qkv, cfg, variant, pre):
    """Attention + norm-gate epilogue -> og [T, Q_SIZE]."""
    if variant is not None:
        ns = _attn_splits(cfg, data, variant, st["nsm"])
        sk = ATTN_SK.get(_attn_class(cfg), 0)
        if sk and variant in (8, 43) and cfg["batch"] * ((cfg["q_len"] + ATTN_SK_Q_BLOCK - 1) // ATTN_SK_Q_BLOCK) * 2 <= ATTN_SK_MAX_PAIRS:
            ns = -sk
        return _attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=variant, pre=pre)
    # kernel_fn's FA3 path views q8 as FP8 (the reference path in _attn_ref passes the raw uint8 tensor).
    return _attention_norm_gate(data, q8.view(FP8), qkv, cfg)


FMHA_CLASSES = {"slide": True, "full_long": True, "full_short": True, "full_batch": True}
FMHA_EXACT = False  # piecewise-linear exp2 on part of the tiles (P is FP8 e4m3 after it anyway); within the task's precision rules
