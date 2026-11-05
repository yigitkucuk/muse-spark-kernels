def _make_ag():
    import ctypes as C
    import threading

    import torch

    class _Ptr(C.c_void_p):
        def __deepcopy__(self, memo):
            return self

    P, I, SZ = _Ptr, C.c_int, C.c_size_t
    TLS = threading.local()
    CUDA = []

    def ctx_bind(x):
        if getattr(TLS, "device", None) == x.device.index:
            return
        if not CUDA:
            cu = C.CDLL("libcuda.so.1")
            for name, args in (("cuCtxGetCurrent", [C.POINTER(C.c_void_p)]),
                               ("cuDeviceGet", [C.POINTER(C.c_int), C.c_int]),
                               ("cuDevicePrimaryCtxRetain", [C.POINTER(C.c_void_p), C.c_int]),
                               ("cuCtxSetCurrent", [C.c_void_p])):
                fn = getattr(cu, name)
                fn.argtypes, fn.restype = args, C.c_int
            CUDA.append(cu)
        cu = CUDA[0]
        ctx = C.c_void_p()
        cu.cuCtxGetCurrent(C.byref(ctx))
        if not ctx.value:
            dev, primary = C.c_int(), C.c_void_p()
            if cu.cuDeviceGet(C.byref(dev), x.device.index or 0) == 0 and cu.cuDevicePrimaryCtxRetain(C.byref(primary), dev) == 0:
                cu.cuCtxSetCurrent(primary)
        TLS.device = x.device.index

    class Algo(C.Structure):
        _fields_ = [("data", C.c_uint64 * 8)]

    class Heur(C.Structure):
        _fields_ = [("algo", Algo), ("workspaceSize", C.c_size_t), ("state", C.c_int),
                    ("wavesCount", C.c_float), ("reserved", C.c_int * 4)]

    def load_lib():
        for name in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
            try:
                return C.CDLL(name)
            except OSError:
                pass
        raise RuntimeError("no libcublasLt")

    class Lt:
        def __init__(self):
            self.lib = load_lib()
            sigs = {
                "cublasLtCreate": [C.POINTER(P)],
                "cublasLtMatmulDescCreate": [C.POINTER(P), I, I],
                "cublasLtMatmulDescSetAttribute": [P, I, P, SZ],
                "cublasLtMatrixLayoutCreate": [C.POINTER(P), I, C.c_uint64, C.c_uint64, C.c_int64],
                "cublasLtMatrixLayoutSetAttribute": [P, I, P, SZ],
                "cublasLtMatmul": [P] * 12 + [C.POINTER(Algo), P, SZ, P],
                "cublasLtMatmulAlgoInit": [P, I, I, I, I, I, I, I, C.POINTER(Algo)],
                "cublasLtMatmulAlgoConfigSetAttribute": [C.POINTER(Algo), I, P, SZ],
                "cublasLtMatmulAlgoCheck": [P, P, P, P, P, P, C.POINTER(Algo), C.POINTER(Heur)],
            }
            for name, argtypes in sigs.items():
                fn = getattr(self.lib, name)
                fn.argtypes, fn.restype = argtypes, I
            self.handle = P()
            self.call("cublasLtCreate", C.byref(self.handle))
            self.desc = {}
            for mode, transpose in (("tn", 1), ("nn", 0)):
                desc = P()
                self.call("cublasLtMatmulDescCreate", C.byref(desc), 68, 0)
                op = I(transpose)
                self.call("cublasLtMatmulDescSetAttribute", desc, 3, C.byref(op), C.sizeof(op))
                self.desc[mode] = desc
            self.workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
            self.required_workspace = 0
            self.alpha, self.beta = C.c_float(1), C.c_float(0)

        def __deepcopy__(self, memo):
            clone = self.__class__.__new__(self.__class__)
            clone.lib, clone.handle, clone.desc = self.lib, self.handle, dict(self.desc)
            clone.workspace = self.workspace.clone()
            clone.required_workspace = self.required_workspace
            clone.alpha, clone.beta = self.alpha, self.beta
            return clone

        def call(self, name, *args):
            status = getattr(self.lib, name)(*args)
            if status:
                raise RuntimeError(f"{name} status {status}")

        def layouts(self, mode, m, n, k):
            # (rows, cols, ld, batch stride); tn: W [n, k] row-major as a column-major k x n, x as k x m, out as n x m
            specs = ((k, n, k, k * n), (k, m, k, 0), (n, m, n, n))
            if mode == "nn":
                specs = ((n, k, n, n),) + specs[1:]  # W^T [k, n] row-major as a column-major n x k
            elif mode == "rownn":
                specs = ((m, k, k, 0), (k, n, n, 0), (m, n, n, 0))  # row-major x [m, k] @ W^T [k, n] -> [m, n]
            out = []
            for rows, cols, ld, stride in specs:
                desc = P()
                self.call("cublasLtMatrixLayoutCreate", C.byref(desc), 14, rows, cols, ld)
                if mode == "rownn":
                    order = I(1)
                    self.call("cublasLtMatrixLayoutSetAttribute", desc, 1, C.byref(order), C.sizeof(order))
                count, offset = I(1), C.c_int64(stride)
                self.call("cublasLtMatrixLayoutSetAttribute", desc, 5, C.byref(count), C.sizeof(count))
                self.call("cublasLtMatrixLayoutSetAttribute", desc, 6, C.byref(offset), C.sizeof(offset))
                out.append(desc)
            return out

        def plan(self, mode, m, n, k, cfg):
            layouts = self.layouts(mode, m, n, k)
            algo = Algo()
            self.call("cublasLtMatmulAlgoInit", self.handle, 68, 0, 14, 14, 14, 14, 66, C.byref(algo))
            for name, attr, ctype in (("tile", 1, C.c_uint32), ("splitk", 2, C.c_int32), ("red", 3, C.c_uint32),
                                      ("swz", 4, C.c_uint32), ("custom", 5, C.c_uint32), ("stages", 6, C.c_uint32),
                                      ("cluster", 8, C.c_uint16)):
                value = ctype(cfg[name])
                status = self.lib.cublasLtMatmulAlgoConfigSetAttribute(C.byref(algo), attr, C.byref(value), C.sizeof(value))
                if status:
                    raise RuntimeError(f"ag config {name} status {status}")
            result = Heur()
            status = self.lib.cublasLtMatmulAlgoCheck(self.handle, self.desc["nn" if mode == "rownn" else mode], *layouts,
                                                      layouts[2], C.byref(algo), C.byref(result))
            if status:
                raise RuntimeError(f"ag AlgoCheck status {status}")
            self.required_workspace = max(self.required_workspace, int(result.workspaceSize))
            return mode, layouts, algo

        def run(self, x, weight, n, plan):
            ctx_bind(x)
            mode, (a, b, c), algo = plan
            out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
            ap = P(x.data_ptr() if mode == "rownn" else weight.data_ptr())
            bp = P(weight.data_ptr() if mode == "rownn" else x.data_ptr())
            self.call("cublasLtMatmul", self.handle, self.desc["nn" if mode == "rownn" else mode], C.byref(self.alpha), ap, a,
                      bp, b, C.byref(self.beta), P(out.data_ptr()), c, P(out.data_ptr()), c,
                      C.byref(algo), P(self.workspace.data_ptr()), self.workspace.numel(),
                      P(torch.cuda.current_stream().cuda_stream))
            return out

    def cfg_of(tile, custom, cluster):
        return {"tile": tile, "stages": 35, "splitk": 1, "red": 0, "swz": 0, "custom": custom, "cluster": cluster}

    # (layer, m) -> (layout mode, tile, custom option, cluster)
    PINS = {("qkv_gate", 256): ("tn", 124, 0, 2), ("pooler", 256): ("rownn", 18, 0, 2), ("o", 256): ("tn", 20, 6, 2),
            ("unpooler", 256): ("rownn", 20, 4, 2), ("pooler", 128): ("nn", 15, 4, 2)}
    TRANSPOSED = {("unpooler", 256), ("pooler", 256), ("pooler", 128)}

    def prepare(weights):
        lt = Lt()
        st = {"lt": lt, "plans": {}, "w": {}}
        wt = {}
        for key, (mode, tile, custom, cluster) in PINS.items():
            layer, m = key
            w = weights[layer]
            n, k = w.shape
            try:
                plan = lt.plan(mode, m, n, k, cfg_of(tile, custom, cluster))
                if key in TRANSPOSED:
                    if layer not in wt:
                        wt[layer] = w.t().contiguous()
                    weight = wt[layer]
                else:
                    weight = w
                lt.workspace = torch.empty(max(lt.required_workspace, lt.workspace.numel()), device="cuda", dtype=torch.uint8)
                g = torch.Generator(device=w.device).manual_seed(7 + m)
                x = torch.randn((m, k), generator=g, device=w.device).to(torch.bfloat16)
                got = lt.run(x, weight, n, plan)
                torch.cuda.synchronize()
                ref = x.float() @ w.float().T
                if bool(torch.isfinite(got).all()) and float((got.float() - ref).norm() / ref.norm()) < 1e-2:
                    st["plans"][key] = plan
                    st["w"][key] = weight
                else:
                    print(f"AG pin {key} failed the self-test", flush=True)
            except Exception as error:  # a pin the library rejects just routes that cell back to the default
                print(f"AG pin {key} unavailable: {str(error)[:120]}", flush=True)
        return st

    def kernel_fn(data, state):
        x = data["x"]
        key = (data["config"]["layer"], x.shape[0])
        return state["lt"].run(x, state["w"][key], data["config"]["n"], state["plans"][key])

    DIMS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}

    def extra_plan(st, layer, m):
        """v6: the m = 256 pin's algorithm on the layouts of another m (cublasLtMatmulAlgoCheck decides if it is valid there);
        True when st['plans'][(layer, m)] exists afterwards. Only called from prepare (the workspace may grow)."""
        key = (layer, m)
        if key in st["plans"]:
            return True
        if (layer, 256) not in st["plans"] or (layer, 256) not in PINS or layer not in DIMS:
            return False
        mode, tile, custom, cluster = PINS[(layer, 256)]
        n, k = DIMS[layer]
        try:
            lt = st["lt"]
            plan = lt.plan(mode, m, n, k, cfg_of(tile, custom, cluster))
            if lt.required_workspace > lt.workspace.numel():
                lt.workspace = torch.empty(lt.required_workspace, device="cuda", dtype=torch.uint8)
        except Exception:
            return False
        st["plans"][key] = plan
        st["w"][key] = st["w"][(layer, 256)]
        return True

    return {"prepare": prepare, "kernel_fn": kernel_fn, "has": lambda state, key: key in state["plans"], "extra_plan": extra_plan}


