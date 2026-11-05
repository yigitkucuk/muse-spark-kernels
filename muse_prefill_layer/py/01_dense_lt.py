def _lt_pin_index(lt, pid, pin):
    """LT_PIN entry -> heuristic index: None (torch.mm), an int index (legacy), or a signature tuple
    (tile id, split-K, reduction scheme, custom option) matched against each heuristic result's config
    read back from cuBLASLt; returns -1 when no result matches (the caller then tunes as usual)."""
    if pin is None:
        return None
    n = lt.lt_nalgo(pid)
    if isinstance(pin, int):
        return pin if pin < n else -1
    for a in range(n):
        info = list(lt.lt_info(pid, a))
        if (info[0], info[1], info[6], info[8]) == tuple(pin):
            return a
    return -1


def _lt_rel_err(out, ref):
    """Max relative error of a candidate GEMM output against the torch.mm reference."""
    return float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6))


LT_PIN2_RACE = 0.997  # a sweep pin is kept only if it measures this much faster than torch.mm on the machine running prepare


def _try_lt2(st, T, name, x, w, out, ref, ref_fn, t_ref, f32):
    """First LT_PIN2 candidate for (T, name) that this cuBLASLt accepts, matches torch.mm numerically and beats it in an
    alternating cold-L2 race; ("lt2", handle) or None (the heuristic pins / tuning then apply). Never raises."""
    cands = LT_PIN2.get((T, name))
    if not cands:
        return None
    lt = st["lt"]
    wn = LT_GEMMS[name][0]
    for lay, cfg in cands:
        try:
            wt = None
            if lay & 2:
                wt = st["lt2_wt"].get(wn) if st.get("lt2_wt") and wn in st["lt2_wt"] else w.t().contiguous()
            h = lt.lt2_add(T, w.shape[0], w.shape[1], f32, lay, cfg, st["lt_ws"].numel())
            if h < 0:
                print(f"LT2 T={T} {name}: lay {lay} cfg {cfg} not offered", flush=True)
                continue
            run = lambda: lt.lt2_run(h, x, w, wt if wt is not None else w, out, st["lt_ws"])
            run()
            torch.cuda.synchronize()
            err = _lt_rel_err(out, ref)
            if not err < LT_ALGO_TOL:
                print(f"LT2 T={T} {name}: lay {lay} cfg {cfg} mismatch {err:.3g}", flush=True)
                continue
            ta, tb = [], []
            for _ in range(3):
                ta.append(_time(run, reps=11))
                tb.append(_time(ref_fn, reps=11))
            a, b = min(ta), min(tb)
            print(f"LT2 T={T} {name}: lay {lay} cfg {cfg}: {a:.1f}us vs torch.mm {b:.1f}us (x{b / a:.3f})", flush=True)
            if a < b * LT_PIN2_RACE:
                if wt is not None:
                    st.setdefault("lt2_wt", {})[wn] = wt
                return ("lt2", h)
        except Exception as exc:  # noqa: BLE001
            print(f"LT2 T={T} {name}: lay {lay} failed: {repr(exc)[:200]}", flush=True)
    return None


def _tune_lt(st):
    """Time cuBLASLt's heuristic algorithms for each dense GEMM at the scored token counts; keep one
    only where it beats torch.mm."""
    lt = st["lt"]
    dev = "cuda"
    choice = {}
    for T in LT_TOKENS:
        for name, (wn, f32) in LT_GEMMS.items():
            w = st[wn]
            N, K = w.shape
            x = (torch.randn((T, K), device=dev) * 0.5).to(torch.bfloat16)
            odt = torch.float32 if f32 else torch.bfloat16
            ref_fn = (lambda: torch.mm(x, w.t(), out_dtype=torch.float32)) if f32 else (lambda: torch.mm(x, w.t()))
            ref = ref_fn().float()
            t_ref = _time(ref_fn, reps=10)
            if name == "router" and not ROUTER_LT:  # (the split-K pick steals the concurrent pooler's SMs)
                del x
                continue
            out = torch.empty((T, N), dtype=odt, device=dev)
            if name == "pooler" and POOL_SM_RESERVE.get(T, 0) > 0:
                pid = lt.lt_plan_sm(T, N, K, f32, st["lt_ws"].numel(), 4, st["nsm"] - POOL_SM_RESERVE[T])
                if lt.lt_nalgo(pid) > 0:
                    lt.lt_run(pid, 0, x, w, out, st["lt_ws"])
                    torch.cuda.synchronize()
                    if _lt_rel_err(out, ref) < LT_ALGO_TOL:
                        choice[(T, name)] = (pid, 0)
                        t = _time(lambda: lt.lt_run(pid, 0, x, w, out, st["lt_ws"]), reps=10)
                        print(f"LT T={T} pooler sized for {st['nsm'] - POOL_SM_RESERVE[T]} SMs: "
                              f"{t:.1f}us alone (torch.mm {t_ref:.1f}us)", flush=True)
                        del x, out
                        continue
            c2 = _try_lt2(st, T, name, x, w, out, ref, ref_fn, t_ref, f32)
            if c2 is not None:
                choice[(T, name)] = c2
                del x, out
                continue
            pid = lt.lt_plan(T, N, K, f32, st["lt_ws"].numel(), 32)
            if (T, name) in LT_PIN:  # a fixed pick: no prepare-time race between near-equal algorithms
                a = _lt_pin_index(lt, pid, LT_PIN[(T, name)])
                if a != -1:  # -1: no heuristic result has the pinned signature -> time them as usual
                    if a is not None:
                        choice[(T, name)] = (pid, a)
                    print(f"LT T={T} {name}: pinned {LT_PIN[(T, name)]} -> algo {a}", flush=True)
                    del x, out
                    continue
                print(f"LT T={T} {name}: pinned signature {LT_PIN[(T, name)]} not offered; tuning instead", flush=True)
            best = (t_ref, None)
            for a in range(lt.lt_nalgo(pid)):
                try:
                    lt.lt_run(pid, a, x, w, out, st["lt_ws"])
                    torch.cuda.synchronize()
                    if not _lt_rel_err(out, ref) < LT_ALGO_TOL:
                        continue
                    t = _time(lambda: lt.lt_run(pid, a, x, w, out, st["lt_ws"]), reps=10)
                except Exception:  # noqa: BLE001
                    continue
                if t < best[0] * LT_MIN_SPEEDUP:
                    best = (t, a)
            if best[1] is not None:
                choice[(T, name)] = (pid, best[1])
            print(f"LT tune T={T} {name}: torch.mm {t_ref:.1f}us, best {best[0]:.1f}us "
                  f"algo {best[1]} of {lt.lt_nalgo(pid)}", flush=True)
            del x, out
    torch.cuda.empty_cache()
    return choice


def _init_lt(st, dev):
    """Build the cuBLASLt extension and tune the dense GEMMs; None choice on any failure."""
    try:
        st["lt"] = _build_lt()
        st["lt_ws"] = torch.empty(LT_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
        st["lt_ws_side"] = torch.empty(LT_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
        return _tune_lt(st)
    except Exception as exc:  # noqa: BLE001
        print("LT unavailable:", repr(exc)[:300], flush=True)
        return None


def _mm(st, x, name, side=False):
    """x [T, K] @ W[N, K]^T with the tuned cuBLASLt algorithm for this shape, else torch.mm."""
    T = x.shape[0]
    choice = st["lt_choice"].get((T, name)) if st.get("lt_choice") else None
    wn, f32 = LT_GEMMS[name]
    w = st[wn]
    if choice is None:
        return torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
    out = torch.empty((T, w.shape[0]), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    if choice[0] == "lt2":
        st["lt"].lt2_run(choice[1], x, w, st.get("lt2_wt", {}).get(wn, w), out, st["lt_ws_side"] if side else st["lt_ws"])
        return out
    st["lt"].lt_run(choice[0], choice[1], x, w, out, st["lt_ws_side"] if side else st["lt_ws"])
    return out


# --------------------------------------------------------------------------------------
# Weight repack: NVFP4 expert weights into the fused kernel's layout.
# --------------------------------------------------------------------------------------


