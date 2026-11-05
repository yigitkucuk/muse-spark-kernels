def _calib_bias(st, dev, g, log=False):
    bias0 = st["bias"].clone()
    try:
        return _rb_prior_calib_bias(st, dev, g, log)
    except BaseException:
        st["bias"].copy_(bias0)
        raise


_rb_prior_prepare = prepare


def prepare(weights):
    global _MOE, _MOE_TINY, _MOE_TINY2, _MOE_MID, _MOE_B4, _PF7
    _MOE = _MOE_TINY = _MOE_TINY2 = _MOE_MID = _MOE_B4 = None
    _PF7 = None
    st = _rb_prior_prepare(weights)
    st["ma_sfx"] = tuple(_MA_SFX)
    st["ma_sc6w"] = bool(_MA_SC6W)
    return st


_rb_prior_kernel_fn = kernel_fn


def kernel_fn(data, state):
    global _MA_SFX, _MA_SC6W
    _MA_SFX = state.get("ma_sfx", _MA_SFX)
    _MA_SC6W = state.get("ma_sc6w", _MA_SC6W)
    return _rb_prior_kernel_fn(data, state)


# ==============================================================================================
# Fused pooler + router (T <= 32) without the router's last-arriver handshake.
# Today every CTA of the fused kernel writes its quarter-K router partial to part[rt][rq][tok][8], does an
# atom.add.release.gpu on rcnt[rt] and waits for the answer, and the 4th arriver adds the four partials in quarter order and
# writes the logits. Here the handshake is gone: the route kernel (_gle_route at T = 1, _sl_tile_route at T = 8 / 16 / 32) adds
# the four partials itself in the same order, ((p0 + p1) + p2) + p3, so the logits are bit-identical.
# ==============================================================================================
import re as _nr_re
import threading as _nr_threading
import time as _nr_time

_NR_PART = {}      # data_ptr of the (now unwritten) logits tensor -> the pooler's partial buffer, consumed by the route wrapper
_NR = {}           # variant extensions
_NR_ERR = {}
_NR_THREADS = []


