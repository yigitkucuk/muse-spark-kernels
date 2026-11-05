_TABLE = {'o:1': 'swapab', 'o:128': 'nb', 'o:16': 'streamk', 'o:256': 'ag', 'o:32': 'swapab', 'o:8': 'streamk', 'pooler:1': 'swapab', 'pooler:128': 'streamk', 'pooler:16': 'streamk', 'pooler:256': 'ag', 'pooler:32': 'swapab', 'pooler:8': 'streamk', 'qkv_gate:1': 'swapab', 'qkv_gate:128': 'streamk', 'qkv_gate:16': 'swapab', 'qkv_gate:256': 'ag', 'qkv_gate:32': 'streamk', 'qkv_gate:8': 'streamk', 'router:1': 'swapab', 'router:128': 'swapab', 'router:16': 'nb', 'router:256': 'swapab', 'router:32': 'nb', 'router:8': 'nb', 'unpooler:1': 'swapab', 'unpooler:128': 'streamk', 'unpooler:16': 'swapab', 'unpooler:256': 'ag', 'unpooler:32': 'swapab', 'unpooler:8': 'streamk'}
_DEFAULT = 'streamk'
_K = {'streamk': _make_streamk(), 'swapab': _make_swapab(), 'nb': _make_nb(), 'ag': _make_ag()}


import math as _v6_math
import time as _v6_time

import torch as _v6_torch

# ======================================================================================================================
V6_ON = True
V6_BUDGET_S = 60.0   # timing + validation budget of the route build (v5's prepare is ~72 s; the limit is 300 s)
V6_GAIN = 0.97       # a candidate must run in under 97% of the default's time at that m (cold-L2 graph replays) to earn the route
V6_REPS = 14
_V6_LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
_V6_N = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
_V6_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
_V6_RANGES = ((9, 15), (17, 31), (33, 127), (129, 255), (257, 384), (2, 7))   # order = priority if the budget runs out
_V6_DETAIL = (20, 28, 33, 40, 48, 64, 80, 100, 127, 129, 160, 200, 255, 257, 300, 384)   # m whose candidate times are logged
_V6_LOG = []


def _v6_say(text):
    _V6_LOG.append(text)


def _v6_n(layer, m):
    return 1024 if (layer == "unpooler" and m <= 8) else _V6_N[layer]


def _v6_up(m):
    return 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0


def _v6_cands(layer, m, state):
    """[(tag, route)] for an off-pin (layer, m): the default first, then the other ways to run it."""
    out = [("def", ("def",))]
    up, n = _v6_up(m), _v6_n(layer, m)
    if up and up != m:
        nm = _TABLE.get("%s:%d" % (layer, up))
        if nm in ("swapab", "nb"):  # the backend pinned at the next bucket up takes any m below its pin
            out.append((nm[0] + str(up), ("pin", nm)))
    if layer == "o" and m > 16 and all(r != ("pin", "nb") for _, r in out):
        out.append(("nbol", ("pin", "nb")))
    if m > 8:
        for key, p in sorted(state["streamk"].get("rq", {}).items()):
            if isinstance(key, tuple) and key[0] == layer and key[1] > m and p["n"] == n:
                out.append(("rq%d" % key[1], ("rq", key[1])))
    if layer != "router" and m > 16 and _K["ag"]["extra_plan"](state["ag"], layer, m):
        out.append(("ag", ("ag",)))
    if layer != "router" and m > 8 and (layer + "_t") in state["streamk"]:
        out.append(("tT", ("tT",)))   # cuBLAS on the K-major copy of W: another heuristic pick
    return out


def _v6_call(route, data, state):
    kind = route[0]
    if kind == "def":
        return _K[_DEFAULT]["kernel_fn"](data, state[_DEFAULT])
    if kind == "pin":
        return _K[route[1]]["kernel_fn"](data, state[route[1]])
    if kind == "ag":
        return _K["ag"]["kernel_fn"](data, state["ag"])
    cw, x = state["streamk"], data["x"]
    if kind == "tT":
        return _v6_torch.mm(x, cw[data["config"]["layer"] + "_t"])
    p = cw["rq"][(data["config"]["layer"], route[1])]
    out = _v6_torch.empty((x.shape[0], p["n"]), dtype=_v6_torch.bfloat16, device=x.device)
    r = _K["streamk"]["rqcall"](cw, p, x.contiguous(), out)
    if r != 0:
        raise RuntimeError("rq launch %d" % r)
    return out


def _v6_x(m, K, dev, gen, outliers):
    """the task's input: standard normal bf16 (outliers: 1% of the channels x30) with a one-hot probe row at the first and last row"""
    x = _v6_torch.randn((m, K), generator=gen, device=dev)
    if outliers:
        x[:, _v6_torch.randperm(K, generator=gen, device=dev)[: max(1, K // 100)]] *= 30
    for j, r in enumerate(sorted({0, m - 1})):
        x[r] = 0
        x[r, (r * 977 + 31 * j + 5) % K] = 64.0 if j == 0 else -32.0
    return x.to(_v6_torch.bfloat16)


def _v6_ok(route, layer, m, state, ref_w, gen, dev):
    """the route's output at (layer, m) on normal and outlier inputs: shape, dtype, rel L2 and the probe rows against the reference"""
    K, n = _V6_K[layer], _v6_n(layer, m)
    router = layer == "router"
    tol, ptol = (3e-5, 3e-4) if router else (6e-3, 1e-3)
    rows = sorted({0, m - 1})
    try:
        for outliers in (False, True):
            x = _v6_x(m, K, dev, gen, outliers)
            out = _v6_call(route, {"x": x, "config": {"m": m, "n": n, "k": K, "layer": layer}}, state)
            if tuple(out.shape) != (m, n) or out.dtype != (_v6_torch.float32 if router else _v6_torch.bfloat16):
                return False
            w = ref_w[layer][:n]
            ref = (x.double() @ w.T) if router else (x.float() @ w.T)
            o = out.double() if router else out.float()
            err = float((o - ref).norm() / ref.norm())
            pr = float(((o[rows] - ref[rows]).norm(dim=1) / ref[rows].norm(dim=1)).max())
            if not (err < tol and pr < ptol):
                _v6_say("V6|bad|%s m%d %s err %.2e probe %.2e" % (layer, m, route[0], err, pr))
                return False
        return True
    except Exception as error:  # noqa: BLE001  (a launch failure just rules the candidate out)
        _v6_say("V6|exc|%s m%d %s %s" % (layer, m, route[0], str(error).replace("\n", " ")[:100]))
        return False


def _v6_bench(routes, data, state, flush, sink, reps):
    """median cold-L2 graph-replay time (us) of every route (None when it fails), replays interleaved in alternating order"""
    gs = []
    for r in routes:
        try:
            _v6_call(r, data, state)
            _v6_torch.cuda.synchronize()
            g = _v6_torch.cuda.CUDAGraph()
            with _v6_torch.cuda.graph(g):
                _v6_call(r, data, state)
            gs.append(g)
        except Exception as error:  # noqa: BLE001
            _v6_say("V6|exc|bench %s %s" % (r[0], str(error).replace("\n", " ")[:100]))
            gs.append(None)
    live = [i for i, g in enumerate(gs) if g is not None]
    ev = [(_v6_torch.cuda.Event(enable_timing=True), _v6_torch.cuda.Event(enable_timing=True)) for _ in gs]
    ts = [[] for _ in gs]
    for rep in range(reps + 3):
        for i in (live if rep % 2 == 0 else live[::-1]):
            _v6_torch.sum(flush, dim=0, out=sink)
            ev[i][0].record()
            gs[i].replay()
            ev[i][1].record()
            _v6_torch.cuda.synchronize()
            if rep >= 3:
                ts[i].append(ev[i][0].elapsed_time(ev[i][1]) * 1e3)
    del gs
    return [sorted(v)[len(v) // 2] if v else None for v in ts]


def _v6_routes(state, weights):
    t0 = _v6_time.perf_counter()
    dev = weights["o"].device
    gen = _v6_torch.Generator(device=dev).manual_seed(6061)
    ref_w = {l: (weights[l].double() if l == "router" else weights[l].float()) for l in _V6_LAYERS}
    flush = _v6_torch.zeros(65536, 1024, dtype=_v6_torch.float32, device=dev)   # 256 MB read sweep (5x the L2) before every replay
    sink = _v6_torch.empty(1024, dtype=_v6_torch.float32, device=dev)
    routes, chosen, cells, skipped = {}, {l: {} for l in _V6_LAYERS}, 0, 0
    for lo, hi in _V6_RANGES:
        for layer in _V6_LAYERS:
            for m in range(lo, hi + 1):
                if "%s:%d" % (layer, m) in _TABLE:
                    continue
                if _v6_time.perf_counter() - t0 > V6_BUDGET_S:
                    skipped += 1
                    continue
                try:
                    cands = _v6_cands(layer, m, state)
                    if len(cands) < 2:
                        continue
                    K, n = _V6_K[layer], _v6_n(layer, m)
                    x = _v6_x(m, K, dev, gen, False)
                    data = {"x": x, "config": {"m": m, "n": n, "k": K, "layer": layer}}
                    ts = _v6_bench([r for _, r in cands], data, state, flush, sink, V6_REPS)
                    cells += 1
                    d = ts[0]
                    if m in _V6_DETAIL:
                        _v6_say("V6|D|%s|%d|def %.1f %s" % (layer, m, d or -1, " ".join(
                            "%s %s" % (c[0], "%.2f" % (t / d) if (t and d) else "x") for c, t in zip(cands[1:], ts[1:]))))
                    if not d:
                        continue
                    for t, tag, route in sorted((t, c[0], c[1]) for c, t in zip(cands[1:], ts[1:]) if t and t < V6_GAIN * d):
                        if not _v6_ok(route, layer, m, state, ref_w, gen, dev):
                            continue
                        if t > 0.9 * d:   # a small win: confirm it with a longer paired run before taking it
                            t2 = _v6_bench([("def",), route], data, state, flush, sink, 3 * V6_REPS)
                            if not (t2[0] and t2[1] and t2[1] < 0.985 * t2[0]):
                                continue
                        routes[(layer, m)] = route
                        chosen[layer][m] = (tag, t / d)
                        break
                except Exception as error:  # noqa: BLE001
                    _v6_say("V6|failed|%s m%d %s" % (layer, m, str(error).replace("\n", " ")[:120]))
    for layer in _V6_LAYERS:
        runs, last = [], None
        for m in sorted(chosen[layer]):
            tag, ratio = chosen[layer][m]
            if last and last[2] == tag and last[1] == m - 1:
                last[1] = m
                last[3].append(ratio)
            else:
                last = [m, m, tag, [ratio]]
                runs.append(last)
        _v6_say("V6|SUM|%s|%s" % (layer, " ".join("%d-%d:%s@%.2f" % (r[0], r[1], r[2], sum(r[3]) / len(r[3])) for r in runs)
                                  or "default everywhere"))
    _v6_say("V6|DONE|%d routes, %d cells timed, %d skipped, %.1f s" % (len(routes), cells, skipped, _v6_time.perf_counter() - t0))
    del ref_w, flush
    _v6_torch.cuda.empty_cache()
    return routes


def prepare(weights):
    import threading
    warm = []
    if _PRUNE:
        for k in _K.values():
            if k.get("warm"):
                def _bg(f=k["warm"]):
                    try:
                        f()
                    except Exception:
                        pass
                t = threading.Thread(target=_bg, daemon=True)
                t.start()
                warm.append(t)
    out = {}
    for name, k in _K.items():
        if name == "swapab":
            for t in warm:
                t.join()
        out[name] = k["prepare"](weights)
    out["routes"] = {}
    if V6_ON:
        try:
            out["routes"] = _v6_routes(out, weights)
        except BaseException as error:  # noqa: BLE001  (any failure leaves the v5 behaviour)
            _v6_say("V6|FAILED|%s: %s" % (type(error).__name__, str(error).replace("\n", " ")[:160]))
            out["routes"] = {}
        print("\n".join(_V6_LOG), flush=True)
    return out


def kernel_fn(data, state):
    cfg = data["config"]
    name = _TABLE.get(f"{cfg['layer']}:{cfg['m']}")
    if name is None:  # off the scored token counts: the v6 route when prepare() found one, else the v5 default
        route = state.get("routes", {}).get((cfg["layer"], cfg["m"]))
        if route is not None:
            return _v6_call(route, data, state)
        name = _DEFAULT
    if name == "ag" and not _K["ag"]["has"](state["ag"], (cfg["layer"], cfg["m"])):
        name = _DEFAULT
    return _K[name]["kernel_fn"](data, state[name])

import ctypes as _CtxC
import threading as _CtxThreading

_CTX_TLS = _CtxThreading.local()
_CTX_API = None


def _ctx_api():
    global _CTX_API
    if _CTX_API is None:
        api = _CtxC.CDLL("libcuda.so.1")
        for name, args in (
            ("cuCtxGetCurrent", [_CtxC.POINTER(_CtxC.c_void_p)]),
            ("cuCtxGetDevice", [_CtxC.POINTER(_CtxC.c_int)]),
            ("cuDeviceGet", [_CtxC.POINTER(_CtxC.c_int), _CtxC.c_int]),
            ("cuDevicePrimaryCtxRetain", [_CtxC.POINTER(_CtxC.c_void_p), _CtxC.c_int]),
            ("cuCtxSetCurrent", [_CtxC.c_void_p]),
        ):
            f = getattr(api, name)
            f.argtypes, f.restype = args, _CtxC.c_int
        _CTX_API = api
    return _CTX_API


def _ctx_check(status, op):
    if status:
        raise RuntimeError(f"{op} failed with CUDA driver status {status}")


def _ctx_bind(x):
    index = x.device.index if x.device.index is not None else 0
    if getattr(_CTX_TLS, "device", None) == index:
        return
    api = _ctx_api()
    ctx = _CtxC.c_void_p()
    _ctx_check(api.cuCtxGetCurrent(_CtxC.byref(ctx)), "cuCtxGetCurrent")
    cur = _CtxC.c_int(-1)
    if ctx.value:
        _ctx_check(api.cuCtxGetDevice(_CtxC.byref(cur)), "cuCtxGetDevice")
    if not ctx.value or cur.value != index:
        dev, primary = _CtxC.c_int(), _CtxC.c_void_p()
        _ctx_check(api.cuDeviceGet(_CtxC.byref(dev), index), "cuDeviceGet")
        _ctx_check(api.cuDevicePrimaryCtxRetain(_CtxC.byref(primary), dev), "cuDevicePrimaryCtxRetain")
        _ctx_check(api.cuCtxSetCurrent(primary), "cuCtxSetCurrent")
    _CTX_TLS.device = index


_ctx_parent_kernel_fn = kernel_fn


def kernel_fn(data, state):
    _ctx_bind(data["x"])
    return _ctx_parent_kernel_fn(data, state)
