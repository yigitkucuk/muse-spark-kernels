def _df_patch(c):
    s = _dft_prev_patch(c)
    assert s.count(_DFT_OLD) == 1, s.count(_DFT_OLD)
    return s.replace(_DFT_OLD, _DFT_NEW)


assert _MA_ALT_SRCS["moe_helper0df"].count(_DFT_OLD) == 1
_MA_ALT_SRCS["moe_helper0df"] = _MA_ALT_SRCS["moe_helper0df"].replace(_DFT_OLD, _DFT_NEW)


def _df_inter(inter):
    """DF: inter rows [2560, 2688) start (and, after every DF launch, end) as bf16 0xFFFF = not yet written."""
    assert inter.numel() >= (_DFT_ROW0 + 128) * 4096, inter.numel()   # scratch[2] is (EMAX + 2 * WGS_XROWS) * 4096 = 11392 rows, allocated once for every T
    inter[_DFT_ROW0 * 4096:(_DFT_ROW0 + 128) * 4096].view(torch.int16).fill_(-1)
    return inter


import threading as _df8_thr

_DF8 = {"on": True, "th": None, "err": None}
_df8_prev_moe = _moe


def _df8_defs(sfx):
    return tuple(("@moe_helper0df" if x == "@moe_helper0" else x) for x in MOE_TINY2_DEFINES) + tuple(sfx)


def _df8_ok(sfx):
    return bool(_DF8["on"] and "@moe_helper0" in MOE_TINY2_DEFINES and "-DOKC_PRMTSC=1" in tuple(sfx) and "moe_helper0df" in _MA_ALT_SRCS
                and (tuple(sfx) or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES)))


def _df8_build(sfx):
    try:
        _ma_ext(_df8_defs(sfx))
    except BaseException as ex:   # noqa: BLE001
        _DF8["err"] = ex


def _moe(M=None, state=None):
    if state is not None and M is not None and MOE_TINY_MAX_M < M <= MOE_TINY2_MAX_M and _df8_ok(state["ma_sfx"]):
        ext = state["ma_ext"].get("tiny2df")
        if ext is None:
            try:
                th = _DF8["th"]
                if th is not None:
                    th.join()
                ext = _RouterExt(_ma_ext(_df8_defs(state["ma_sfx"])))
                if MOE_HAS_PDL:
                    ext.okc_set_pdl(0)
                state["ma_ext"]["tiny2df"] = ext
            except BaseException as ex:   # noqa: BLE001
                _DF8["on"] = False
                print(f"[df8] off (build): {type(ex).__name__}: {str(ex)[-300:]}", flush=True)
                return _df8_prev_moe(M, state)
        return ext
    return _df8_prev_moe(M, state)


def _df8_selftest(st, dev):
    g = torch.Generator(device=dev).manual_seed(777)
    for T in (5, 8):
        x = torch.randn((T, 4096), generator=g, device=dev).to(torch.bfloat16)
        sc = torch.rand((T, 256), generator=g, device=dev)
        sc[:, :32] += 0.03
        sc[:, 0] += 0.35
        sc[:, 1] += 0.25
        top = sc.topk(8, dim=-1)
        ids = top.indices
        lids = torch.where(ids < 32, ids, torch.full_like(ids, -1)).to(torch.int32).contiguous()
        wts = (top.values / top.values.sum(-1, keepdim=True)).float().contiguous()
        ob = torch.zeros((T, 4096), dtype=torch.bfloat16, device=dev)
        be = _df8_prev_moe(T, st)
        be.okc_run(st["ma_w"], st["ma_scratch"], x, lids, wts, ob, 0, 32, st["nsm"])
        for rep in range(4):
            oc = torch.zeros((T, 4096), dtype=torch.bfloat16, device=dev)
            _moe(T, st).okc_run(st["ma_w"], st["ma_scratch"], x, lids, wts, oc, 0, 32, st["nsm"])
            torch.cuda.synchronize()
            rel = float((oc.float() - ob.float()).norm() / ob.float().norm().clamp(min=1e-20))
            if rep == 0:
                print(f"[df8] selftest T={T}: rel L2 vs the plain TINY2 kernel {rel:.3e}, local routes {int((lids >= 0).sum())}", flush=True)
            if not rel < 2e-3:
                return False
    return True


_df8_prior_prepare = prepare


def prepare(weights):
    try:
        sfx = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
        if _df8_ok(sfx):
            _DF8["th"] = _df8_thr.Thread(target=_df8_build, args=(sfx,), daemon=True)
            _DF8["th"].start()
    except BaseException as ex:   # noqa: BLE001
        print(f"[df8] prebuild not started: {type(ex).__name__}: {str(ex)[-200:]}", flush=True)
    st = _df8_prior_prepare(weights)
    try:
        ok = _DF8["on"] and _df8_ok(st["ma_sfx"]) and _df8_selftest(st, weights["qkv_gate"].device)
        if not ok:
            _DF8["on"] = False
        print(f"[df8] data-flag TINY2 at T=5..8: {'on' if _DF8['on'] else 'off'}", flush=True)
    except BaseException as ex:   # noqa: BLE001
        _DF8["on"] = False
        print(f"[df8] off: {type(ex).__name__}: {str(ex)[-300:]}", flush=True)
    return st


MOEV_BITS = 1081472          # OKC_LAB bitmask baked into the build (128 FP1, 256 PF, 512 SHFALL, 2048 EARLY, 4096 PFALL)
MOEV_SC5W = True
MOEV_TS = (128, 256)              # shapes the LAB times (the lab dict); the candidate itself is active for EVERY M > 32 (the file's main WGS build)
MOEV_MIN_M = 33               # first M that takes the main WGS build in _moe (<= 32: tiny / tiny2 / mid / B4FIN builds, untouched)
MOEV_SELFTEST_TS = (33, 48, 64, 96, 128, 192, 256, 384)   # off-grid and on-grid token counts of the main build (MMAX = 384)
MOEV_EXTRA = ()        # extra -D defines
